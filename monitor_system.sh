#!/bin/bash

# 系统资源实时监测脚本
# 使用方法: ./monitor_system.sh

# 显示初始信息
echo "=========================================="
echo "系统资源实时监测脚本"
echo "=========================================="
echo "按 Ctrl+C 退出"
echo "监控间隔: 3秒 | 无效进程检查: 30分钟"
echo "=========================================="
echo ""

# 初始化上一次采样的计数（CPU/网络/磁盘）
prev_total=0; prev_idle_all=0; prev_iowait=0
prev_rx_bytes=0; prev_tx_bytes=0
prev_rd_sectors=0; prev_wr_sectors=0
have_prev=0

# 无效进程检查计时器（30分钟 = 1800秒，每3秒刷新一次，需要600次循环）
check_interval=600  # 30分钟检查一次
check_counter=0

# 资源告警阈值配置（可根据系统配置调整）
CPU_WARN_THRESHOLD=80      # CPU使用率警告阈值(%)
CPU_CRITICAL_THRESHOLD=95  # CPU使用率严重警告阈值(%)
MEM_WARN_THRESHOLD=80      # 内存使用率警告阈值(%)
MEM_CRITICAL_THRESHOLD=90  # 内存使用率严重警告阈值(%)
DISK_WARN_THRESHOLD=85     # 磁盘使用率警告阈值(%)
DISK_CRITICAL_THRESHOLD=95 # 磁盘使用率严重警告阈值(%)
SWAP_WARN_THRESHOLD=50     # Swap使用率警告阈值(%)
LOAD_WARN_MULTIPLIER=2.0   # 负载警告倍数（相对于CPU核心数）

# 进程监控配置
PROCESS_CPU_WARN=50        # 单个进程CPU使用率警告阈值(%)
PROCESS_MEM_WARN=10        # 单个进程内存使用率警告阈值(%)
ZOMBIE_WARN_COUNT=5        # 僵尸进程警告数量

# 历史数据存储（用于趋势分析）
declare -a cpu_history=()
declare -a mem_history=()
MAX_HISTORY=10  # 保留最近10次采样

# 多磁盘上一轮扇区计数（用于统计SSD/HDD总吞吐）
declare -A prev_rd_map
declare -A prev_wr_map

# 智能识别工作目录和磁盘
# 1. 识别当前工作目录（脚本运行位置或PWD）
WORK_DIR="${PWD:-$(pwd)}"
# 2. 识别用户主目录
USER_HOME="${HOME:-$(eval echo ~$USER)}"
# 3. 识别最大的非系统数据分区（排除 /, /boot, /tmp 等系统分区）
# 优先使用当前工作目录所在分区，否则选择最大的数据分区
if [ -d "$WORK_DIR" ] && df -P "$WORK_DIR" >/dev/null 2>&1; then
    MONITOR_DISK=$(df -P "$WORK_DIR" | tail -1 | awk '{print $6}')
else
    # 选择最大的非系统分区
    MONITOR_DISK=$(df -P | awk 'NR>1 && $6 !~ /^\/(boot|tmp|var|usr|dev|proc|sys|run)$/ && $6 != "/" {
        size=$2; mount=$6
        if (size > max || max == "") { max=size; max_mount=mount }
    } END {print max_mount}')
fi

# 识别监控磁盘所在块设备（用于磁盘吞吐监测）
if [ -n "$MONITOR_DISK" ] && [ -d "$MONITOR_DISK" ]; then
    disk_dev=$(df -P "$MONITOR_DISK" 2>/dev/null | tail -1 | awk '{print $1}')
    disk_name=${disk_dev##*/}
else
    disk_name=""
fi

# 识别用户工作路径模式（用于进程过滤）
# 提取工作目录和主目录的路径前缀
WORK_DIR_PREFIX=$(echo "$WORK_DIR" | sed 's|/[^/]*$||')
USER_HOME_PREFIX=$(echo "$USER_HOME" | sed 's|/[^/]*$||')

# 获取当前用户名（用于进程过滤）
CURRENT_USER="${USER:-$(whoami)}"

# 获取CPU核心数（用于负载评估）
CPU_CORES=$(nproc 2>/dev/null || echo 1)

# 资源告警函数
check_resource_alerts() {
    local alerts=""
    local critical_alerts=""
    
    # CPU告警
    if (( $(echo "$cpu_pct > $CPU_CRITICAL_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
        critical_alerts="${critical_alerts}[严重] CPU使用率: ${cpu_pct}% (阈值: ${CPU_CRITICAL_THRESHOLD}%)\n"
    elif (( $(echo "$cpu_pct > $CPU_WARN_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
        alerts="${alerts}[警告] CPU使用率: ${cpu_pct}% (阈值: ${CPU_WARN_THRESHOLD}%)\n"
    fi
    
    # 内存告警
    if (( $(echo "$mem_percent > $MEM_CRITICAL_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
        critical_alerts="${critical_alerts}[严重] 内存使用率: ${mem_percent}% (阈值: ${MEM_CRITICAL_THRESHOLD}%)\n"
    elif (( $(echo "$mem_percent > $MEM_WARN_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
        alerts="${alerts}[警告] 内存使用率: ${mem_percent}% (阈值: ${MEM_WARN_THRESHOLD}%)\n"
    fi
    
    # 磁盘告警
    if [ -n "$disk_percent" ]; then
        disk_pct=$(echo "$disk_percent" | sed 's/%//')
        if (( $(echo "$disk_pct > $DISK_CRITICAL_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
            critical_alerts="${critical_alerts}[严重] 磁盘使用率: ${disk_percent} (阈值: ${DISK_CRITICAL_THRESHOLD}%)\n"
        elif (( $(echo "$disk_pct > $DISK_WARN_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
            alerts="${alerts}[警告] 磁盘使用率: ${disk_percent} (阈值: ${DISK_WARN_THRESHOLD}%)\n"
        fi
    fi
    
    # Swap告警
    swap_info=$(free | awk 'NR==3{if($2>0) printf "%.1f", $3*100/$2; else print "0"}')
    if (( $(echo "$swap_info > $SWAP_WARN_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
        alerts="${alerts}[警告] Swap使用率: ${swap_info}% (阈值: ${SWAP_WARN_THRESHOLD}%)\n"
    fi
    
    # 负载告警
    load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    load_threshold=$(awk -v cores=$CPU_CORES -v mult=$LOAD_WARN_MULTIPLIER 'BEGIN{printf "%.2f", cores*mult}')
    if (( $(echo "$load_1min > $load_threshold" | bc -l 2>/dev/null || echo 0) )); then
        alerts="${alerts}[警告] 系统负载: ${load_1min} (CPU核心数: ${CPU_CORES}, 阈值: ${load_threshold})\n"
    fi
    
    # 显示告警
    if [ -n "$critical_alerts" ]; then
        echo ""
        echo "=========================================="
        echo -e "严重告警:\n${critical_alerts}"
        echo "=========================================="
    fi
    if [ -n "$alerts" ]; then
        echo ""
        echo "------------------------------------------"
        echo -e "警告:\n${alerts}"
        echo "------------------------------------------"
    fi
}

# 检查异常进程（仅当前用户）
check_abnormal_processes() {
    local abnormal_procs=""
    
    # 检查高CPU占用进程（仅当前用户）
    high_cpu=$(ps -eo pid,user,%cpu,%mem,rss,cmd --no-headers --sort=-%cpu | \
        awk -v threshold=$PROCESS_CPU_WARN -v current_user="$CURRENT_USER" \
        '$2 == current_user && $3 > threshold && $3 < 100 {print $0}' | head -3)
    
    # 检查高内存占用进程（仅当前用户）
    high_mem=$(ps -eo pid,user,%mem,%cpu,rss,cmd --no-headers --sort=-%mem | \
        awk -v threshold=$PROCESS_MEM_WARN -v current_user="$CURRENT_USER" \
        '$2 == current_user && $2 > threshold {print $0}' | head -3)
    
    # 检查僵尸进程（仅当前用户）
    zombie_count=$(ps aux | awk -v current_user="$CURRENT_USER" \
        '$1 == current_user && $8 ~ /^Z/ {count++} END {print count+0}')
    
    if [ -n "$high_cpu" ] || [ -n "$high_mem" ] || [ "$zombie_count" -gt 0 ]; then
        echo ""
        echo "异常进程监控(用户: $CURRENT_USER):"
        if [ -n "$high_cpu" ]; then
            echo "  高CPU占用进程:"
            echo "$high_cpu" | while read pid user cpu mem rss cmd; do
                cmd_short=$(echo "$cmd" | cut -c1-50)
                printf "    PID %-6s CPU: %5s%% MEM: %5s%% %s\n" "$pid" "$cpu" "$mem" "$cmd_short"
            done
        fi
        if [ -n "$high_mem" ]; then
            echo "  高内存占用进程:"
            echo "$high_mem" | while read pid user mem cpu rss cmd; do
                cmd_short=$(echo "$cmd" | cut -c1-50)
                printf "    PID %-6s MEM: %5s%% CPU: %5s%% %s\n" "$pid" "$mem" "$cpu" "$cmd_short"
            done
        fi
        if [ "$zombie_count" -gt 0 ]; then
            if [ "$zombie_count" -ge $ZOMBIE_WARN_COUNT ]; then
                echo "  [警告] 发现当前用户的 ${zombie_count} 个僵尸进程"
            else
                echo "  发现当前用户的 ${zombie_count} 个僵尸进程"
            fi
        fi
    fi
}

# 计算资源使用趋势
calculate_trend() {
    local current_value=$1
    local history_name=$2
    local trend=""
    
    # 使用全局数组变量
    if [ "$history_name" = "cpu_history" ]; then
        cpu_history+=($current_value)
        local history_size=${#cpu_history[@]}
        if [ "$history_size" -gt $MAX_HISTORY ]; then
            cpu_history=("${cpu_history[@]:1}")
            history_size=$MAX_HISTORY
        fi
        if [ "$history_size" -ge 3 ]; then
            local first=${cpu_history[0]}
            local last=${cpu_history[$((history_size-1))]}
            local diff=$(awk -v f=$first -v l=$last 'BEGIN{printf "%.1f", l-f}')
            if (( $(echo "$diff > 5" | bc -l 2>/dev/null || echo 0) )); then
                trend="↑"
            elif (( $(echo "$diff < -5" | bc -l 2>/dev/null || echo 0) )); then
                trend="↓"
            else
                trend="→"
            fi
        fi
    elif [ "$history_name" = "mem_history" ]; then
        mem_history+=($current_value)
        local history_size=${#mem_history[@]}
        if [ "$history_size" -gt $MAX_HISTORY ]; then
            mem_history=("${mem_history[@]:1}")
            history_size=$MAX_HISTORY
        fi
        if [ "$history_size" -ge 3 ]; then
            local first=${mem_history[0]}
            local last=${mem_history[$((history_size-1))]}
            local diff=$(awk -v f=$first -v l=$last 'BEGIN{printf "%.1f", l-f}')
            if (( $(echo "$diff > 5" | bc -l 2>/dev/null || echo 0) )); then
                trend="↑"
            elif (( $(echo "$diff < -5" | bc -l 2>/dev/null || echo 0) )); then
                trend="↓"
            else
                trend="→"
            fi
        fi
    fi
    
    echo "$trend"
}

# 检查无效进程的函数（仅当前用户）
check_invalid_processes() {
    # 查找运行时间超过1个月（30天）且CPU使用率低于0.1%的进程
    # 仅检查当前用户的进程，排除系统进程和内核线程
    # 将路径信息传递给awk
    export WORK_DIR_PREFIX USER_HOME_PREFIX CURRENT_USER
    invalid_pids=$(ps -eo pid,user,etime,stat,%cpu,%mem,rss,cmd --no-headers | \
        awk -v work_pattern="$WORK_DIR_PREFIX" -v home_pattern="$USER_HOME_PREFIX" -v current_user="$CURRENT_USER" '{
            # 检查是否为当前用户的进程
            if ($2 != current_user) next
            
            # 解析运行时间（格式: DD-HH:MM:SS 或 HH:MM:SS）
            # 注意：由于添加了user字段，etime现在是第3列，stat是第4列，%cpu是第5列
            split($3, time_parts, "-")
            days = 0
            if (length(time_parts) == 2) {
                days = time_parts[1] + 0
            }
            # 检查条件：运行时间>=30天，CPU<0.1%，不是内核线程（stat不以[开头）
            if (days >= 30 && $5 < 0.1 && $4 !~ /^\[/) {
                # 获取命令（从第8列开始到末尾，因为添加了user字段）
                cmd = ""
                for (i = 8; i <= NF; i++) {
                    if (i > 8) cmd = cmd " "
                    cmd = cmd $i
                }
                # 排除系统关键进程和内核线程
                # 重点关注用户工作目录下的进程（如Python分析脚本）
                is_system_process = 0
                is_user_process = 0
                
                # 检查是否是用户工作目录下的进程
                # 使用动态识别的路径模式
                if (work_pattern != "" && cmd ~ work_pattern) {
                    is_user_process = 1
                } else if (home_pattern != "" && cmd ~ home_pattern) {
                    is_user_process = 1
                } else if (cmd ~ /calculate\// || cmd ~ /MMPE\// || \
                    cmd ~ /python.*\.py/ || cmd ~ /\.py$/) {
                    is_user_process = 1
                }
                
                # 排除系统进程
                if (cmd ~ /(systemd|kthreadd|ksoftirqd|migration|rcu_|watchdog|kworker|\[|lvmetad|multipathd|udevd|journald|lsmd|abrt|accounts-daemon|udisksd|rtkit|avahi|polkitd|firewalld|ksmtuned|gnome|dbus|pulseaudio|ibus|Xorg|X :)/ || \
                    cmd ~ /^\/usr\/lib\/systemd/ || \
                    cmd ~ /^\/sbin\// || \
                    cmd ~ /^\/usr\/sbin\// || \
                    cmd ~ /^\/usr\/libexec\// || \
                    cmd ~ /^\/usr\/lib\// || \
                    cmd ~ /^\/usr\/bin\/(python2|X|gnome|dbus|pulseaudio|ibus)/) {
                    is_system_process = 1
                }
                
                # 只显示用户进程（排除系统进程）
                # 输出：PID, 运行时间, CPU%, MEM%, RSS, 命令
                if (is_system_process == 0 && is_user_process == 1) {
                    print $1, $3, $5, $6, $7, cmd
                }
            }
        }')
    
    if [ -n "$invalid_pids" ]; then
        echo ""
        echo "=========================================="
        echo "警告: 发现当前用户($CURRENT_USER)长时间运行且CPU使用率极低的进程"
        echo "=========================================="
        echo "PID    运行时间      CPU%   MEM%   RSS(MB)   命令"
        echo "$invalid_pids" | while read pid etime cpu mem rss cmd; do
            rss_mb=$(awk -v r=$rss 'BEGIN{printf "%.0f", r/1024}')
            # 截断过长的命令显示
            cmd_short=$(echo "$cmd" | cut -c1-60)
            printf "%-7s %-12s %-6s %-6s %-9s %s\n" "$pid" "$etime" "$cpu" "$mem" "$rss_mb" "$cmd_short"
        done
        
        # 计算总内存占用
        total_rss=$(echo "$invalid_pids" | awk '{sum+=$5} END {printf "%.0f", sum/1024}')
        total_count=$(echo "$invalid_pids" | wc -l)
        echo ""
        echo "总计: ${total_count} 个进程，占用约 ${total_rss} MB 内存"
        echo ""
        echo "是否要终止这些进程? (y/n)"
        read -t 30 -n 1 answer
        echo ""
        
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            echo "正在终止当前用户($CURRENT_USER)的进程..."
            pids_to_kill=$(echo "$invalid_pids" | awk '{print $1}')
            killed_count=0
            for pid in $pids_to_kill; do
                # 双重验证：确保进程属于当前用户
                process_user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
                if [ "$process_user" = "$CURRENT_USER" ]; then
                    if kill $pid 2>/dev/null; then
                        killed_count=$((killed_count + 1))
                    fi
                else
                    echo "  跳过PID $pid (不属于当前用户)"
                fi
            done
            sleep 2
            # 强制终止仍未退出的进程（再次验证用户）
            for pid in $pids_to_kill; do
                process_user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
                if [ "$process_user" = "$CURRENT_USER" ] && ps -p $pid >/dev/null 2>&1; then
                    kill -9 $pid 2>/dev/null
                fi
            done
            echo "已终止当前用户的 ${killed_count} 个进程"
            echo "按任意键继续..."
            read -n 1
        else
            echo "已跳过清理，将在30分钟后再次检查"
            echo "按任意键继续..."
            read -n 1
        fi
    fi
}

while true; do
    # 临时输出文件，整屏一次性渲染，避免逐行闪烁
    tmp_output=$(mktemp)
    perform_check=0
    
    {
        # 显示标题和时间
        echo "=========================================="
        echo "系统资源监控 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""
    
    # CPU使用率（从 /proc/stat 计算，更稳健）
    if [ -r /proc/stat ]; then
        read cpu user nice sys idle iow irq softirq steal guest guest_nice < /proc/stat
        idle_all=$((idle + iow))
        non_idle=$((user + nice + sys + irq + softirq + steal))
        total=$((idle_all + non_idle))
        if [ $have_prev -eq 1 ]; then
            totald=$((total - prev_total))
            idled=$((idle_all - prev_idle_all))
            iowd=$((iow - prev_iowait))
            if [ $totald -gt 0 ]; then
                cpu_pct=$(awk -v t=$totald -v i=$idled 'BEGIN{printf "%.1f", (t-i)/t*100}')
                iow_pct=$(awk -v t=$totald -v w=$iowd 'BEGIN{printf "%.1f", (w<0?0:w)/t*100}')
            else
                cpu_pct=0.0; iow_pct=0.0
            fi
        else
            cpu_pct=0.0; iow_pct=0.0
        fi
        cpu_trend=$(calculate_trend "$cpu_pct" "cpu_history")
        printf "CPU: %s%% %s (iowait %s%%, 核心数: %s)\n" "$cpu_pct" "$cpu_trend" "$iow_pct" "$CPU_CORES"
        prev_total=$total; prev_idle_all=$idle_all; prev_iowait=$iow
    else
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed 's/,/ /g' | awk '{for(i=1;i<=NF;i++){if($i~/%id/){sub("%id,","",$i); idle=$i}}} END{printf "%.1f", 100-idle}')
        cpu_trend=$(calculate_trend "$cpu_usage" "cpu_history")
        echo "CPU: ${cpu_usage}% ${cpu_trend}"
    fi
    
    # 内存使用情况
    mem_info=$(free -h | grep "Mem:")
    mem_used=$(echo $mem_info | awk '{print $3}')
    mem_total=$(echo $mem_info | awk '{print $2}')
    mem_available=$(free -h | grep "Mem:" | awk '{print $7}')
    mem_percent=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    mem_trend=$(calculate_trend "$mem_percent" "mem_history")
    
    # Swap信息
    swap_info=$(free -h | grep "Swap:")
    swap_total=$(echo $swap_info | awk '{print $2}')
    swap_used=$(echo $swap_info | awk '{print $3}')
    swap_percent=$(free | awk 'NR==3{if($2>0) printf "%.1f", $3*100/$2; else print "0"}')
    
    echo "内存: ${mem_used}/${mem_total} (${mem_percent}%) ${mem_trend} | 可用: ${mem_available}"
    if [ "$swap_total" != "0B" ] && [ -n "$swap_total" ]; then
        echo "Swap: ${swap_used}/${swap_total} (${swap_percent}%)"
    fi
    
    # 磁盘使用情况（显示监控的分区）
    if [ -n "$MONITOR_DISK" ] && [ -d "$MONITOR_DISK" ]; then
        disk_info=$(df -h "$MONITOR_DISK" 2>/dev/null | tail -1)
        if [ -n "$disk_info" ]; then
            disk_used=$(echo $disk_info | awk '{print $3}')
            disk_total=$(echo $disk_info | awk '{print $2}')
            disk_percent=$(echo $disk_info | awk '{print $5}')
            echo "磁盘($MONITOR_DISK): ${disk_used}/${disk_total} (${disk_percent})"
            # inode 使用
            if df -i "$MONITOR_DISK" >/dev/null 2>&1; then
                inode_info=$(df -i "$MONITOR_DISK" | tail -1)
                inode_used=$(echo $inode_info | awk '{print $3}')
                inode_total=$(echo $inode_info | awk '{print $2}')
                inode_percent=$(echo $inode_info | awk '{print $5}')
                echo "inode($MONITOR_DISK): ${inode_used}/${inode_total} (${inode_percent})"
            fi
        fi
    fi
    
    # 磁盘吞吐（基于 /proc/diskstats 的读写字节/秒）
    if [ -r /proc/diskstats ]; then
        # 1) 若能识别监控磁盘的设备，则单独展示其吞吐
        if [ -n "$disk_name" ] && grep -qw "$disk_name" /proc/diskstats; then
            line=$(grep -w "$disk_name" /proc/diskstats)
            rd_sec=$(echo "$line" | awk '{print $6}')
            wr_sec=$(echo "$line" | awk '{print $10}')
            if [ $have_prev -eq 1 ]; then
                rd_diff=$((rd_sec - prev_rd_sectors))
                wr_diff=$((wr_sec - prev_wr_sectors))
                r_mb=$(awk -v s=$rd_diff 'BEGIN{printf "%.2f", s*512/1024/1024}')
                w_mb=$(awk -v s=$wr_diff 'BEGIN{printf "%.2f", s*512/1024/1024}')
                echo "挂载设备吞吐(${disk_name}): 读 ${r_mb} MB/s | 写 ${w_mb} MB/s"
            fi
            prev_rd_sectors=$rd_sec; prev_wr_sectors=$wr_sec
        fi

        # 2) 统计所有块设备：按SSD/HDD分类聚合吞吐
        ssd_r_mb=0; ssd_w_mb=0; hdd_r_mb=0; hdd_w_mb=0
        for devpath in /sys/block/*; do
            dev=$(basename "$devpath")
            case "$dev" in
                loop*|ram*|sr* ) continue ;;
            esac
            # 仅统计存在于 /proc/diskstats 的主设备行
            line=$(awk -v d="$dev" '$3==d {print; exit}' /proc/diskstats)
            [ -z "$line" ] && continue
            rd_sec=$(echo "$line" | awk '{print $6}')
            wr_sec=$(echo "$line" | awk '{print $10}')

            prev_r=${prev_rd_map[$dev]:-0}
            prev_w=${prev_wr_map[$dev]:-0}
            if [ $have_prev -eq 1 ]; then
                rd_diff=$((rd_sec - prev_r))
                wr_diff=$((wr_sec - prev_w))
                r_mb=$(awk -v s=$rd_diff 'BEGIN{printf "%.2f", s*512/1024/1024}')
                w_mb=$(awk -v s=$wr_diff 'BEGIN{printf "%.2f", s*512/1024/1024}')
                # 判断是否SSD：/sys/block/<dev>/queue/rotational 为0表示SSD
                rot=1
                if [ -r "$devpath/queue/rotational" ]; then rot=$(cat "$devpath/queue/rotational" 2>/dev/null || echo 1); fi
                if [ "$rot" = "0" ]; then
                    ssd_r_mb=$(awk -v a=$ssd_r_mb -v b=$r_mb 'BEGIN{printf "%.2f", a+b}')
                    ssd_w_mb=$(awk -v a=$ssd_w_mb -v b=$w_mb 'BEGIN{printf "%.2f", a+b}')
                else
                    hdd_r_mb=$(awk -v a=$hdd_r_mb -v b=$r_mb 'BEGIN{printf "%.2f", a+b}')
                    hdd_w_mb=$(awk -v a=$hdd_w_mb -v b=$w_mb 'BEGIN{printf "%.2f", a+b}')
                fi
            fi
            prev_rd_map[$dev]=$rd_sec
            prev_wr_map[$dev]=$wr_sec
        done
        if [ $have_prev -eq 1 ]; then
            # 仅当对应类别存在设备时输出
            if [ "$ssd_r_mb" != "0.00" ] || [ "$ssd_w_mb" != "0.00" ]; then
                echo "SSD总吞吐: 读 ${ssd_r_mb} MB/s | 写 ${ssd_w_mb} MB/s"
            fi
            if [ "$hdd_r_mb" != "0.00" ] || [ "$hdd_w_mb" != "0.00" ]; then
                echo "HDD总吞吐: 读 ${hdd_r_mb} MB/s | 写 ${hdd_w_mb} MB/s"
            fi
        fi
    fi
    
    # 系统负载
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^ //')
    load_1min=$(echo "$load_avg" | awk '{print $1}' | sed 's/,//')
    load_ratio=$(awk -v load=$load_1min -v cores=$CPU_CORES 'BEGIN{printf "%.2f", load/cores}')
    echo "负载(1/5/15): ${load_avg} | 负载/核心比: ${load_ratio}"
    
    # GPU使用情况（如果有NVIDIA GPU）
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU状态:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r index util mem_used mem_total temp; do
            echo "  GPU${index}: ${util}% | ${mem_used}MB/${mem_total}MB | ${temp}°C"
        done
    fi
    
    # 网络吞吐（聚合非lo）
    if [ -r /proc/net/dev ]; then
        rx_bytes=0; tx_bytes=0
        while read -r line; do
            iface=$(echo "$line" | awk -F: '{print $1}' | tr -d ' ')
            stats=$(echo "$line" | awk -F: '{print $2}')
            [ -z "$iface" ] && continue
            [ "$iface" = "lo" ] && continue
            rx=$(echo "$stats" | awk '{print $1}')
            tx=$(echo "$stats" | awk '{print $9}')
            rx_bytes=$((rx_bytes + rx))
            tx_bytes=$((tx_bytes + tx))
        done < <(tail -n +3 /proc/net/dev)
        if [ $have_prev -eq 1 ]; then
            rx_mbps=$(awk -v d=$((rx_bytes - prev_rx_bytes)) 'BEGIN{printf "%.2f", d*8/1024/1024}')
            tx_mbps=$(awk -v d=$((tx_bytes - prev_tx_bytes)) 'BEGIN{printf "%.2f", d*8/1024/1024}')
            echo "网络(总): 下行 ${rx_mbps} Mbps | 上行 ${tx_mbps} Mbps"
        fi
        prev_rx_bytes=$rx_bytes; prev_tx_bytes=$tx_bytes
    fi
    
    # 网络连接统计
    if command -v ss &> /dev/null; then
        tcp_conn=$(ss -s 2>/dev/null | grep TCP | awk '{print $2}' | head -1)
        if [ -n "$tcp_conn" ]; then
            echo "TCP连接数: ${tcp_conn}"
        fi
    elif [ -r /proc/net/sockstat ]; then
        tcp_conn=$(grep TCP /proc/net/sockstat | awk '{print $3}')
        if [ -n "$tcp_conn" ]; then
            echo "TCP连接数: ${tcp_conn}"
        fi
    fi
    
    # 文件描述符使用情况
    if [ -r /proc/sys/fs/file-nr ]; then
        fd_info=$(cat /proc/sys/fs/file-nr)
        fd_used=$(echo $fd_info | awk '{print $1}')
        fd_max=$(cat /proc/sys/fs/file-max 2>/dev/null || echo "N/A")
        if [ "$fd_max" != "N/A" ]; then
            fd_percent=$(awk -v used=$fd_used -v max=$fd_max 'BEGIN{printf "%.1f", used*100/max}')
            echo "文件描述符: ${fd_used}/${fd_max} (${fd_percent}%)"
        else
            echo "文件描述符: ${fd_used} (最大: ${fd_max})"
        fi
    fi

        # 进程统计
        total_procs=$(ps aux | wc -l)
        running_procs=$(ps aux | awk '$8 ~ /^R/ {count++} END {print count+0}')
        sleeping_procs=$(ps aux | awk '$8 ~ /^S/ {count++} END {print count+0}')
        zombie_procs=$(ps aux | awk '$8 ~ /^Z/ {count++} END {print count+0}')
        echo ""
        echo "进程统计: 总计 ${total_procs} | 运行中 ${running_procs} | 睡眠 ${sleeping_procs} | 僵尸 ${zombie_procs}"
        
        # Top进程（内存/CPU）
        echo ""
        echo "Top内存进程(前5):"
        ps -eo pid,user,%mem,%cpu,rss,comm --sort=-%mem | head -n 6 | sed '1d'
        echo ""
        echo "Top CPU进程(前5):"
        ps -eo pid,user,%cpu,%mem,rss,comm --sort=-%cpu | head -n 6 | sed '1d'
        
        # 检查异常进程
        check_abnormal_processes
        
        # 资源告警检查
        check_resource_alerts

        # 每30分钟检查一次无效进程（仅标记，实际执行在缓冲输出后）
        check_counter=$((check_counter + 1))
        if [ $check_counter -ge $check_interval ]; then
            perform_check=1
            check_counter=0  # 重置计数器
        fi
    } > "$tmp_output"

    # 整屏一次性刷新
    clear
    printf '\033[2J\033[H'
    cat "$tmp_output"
    rm -f "$tmp_output"

    # 缓冲输出后再进行需要交互的清理提示，避免提示被缓冲
    if [ $perform_check -eq 1 ]; then
        check_invalid_processes
    fi

    # 等待3秒后刷新
    sleep 3
    have_prev=1
done