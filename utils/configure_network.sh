#!/bin/bash

# 检查是否以root权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：此脚本需要root权限运行，请使用sudo执行"
    exit 1
fi

# 函数：获取有线网络接口
get_wired_interface() {
    # 获取所有网络接口（排除回环接口）
    interfaces=$(ip link show | awk -F': ' '/^[0-9]+: / && !/lo:/ {print $2}')
    
    # 优先查找eth或enp开头的有线接口
    for iface in $interfaces; do
        if [[ $iface == eth* || $iface == enp* ]]; then
            echo "$iface"
            return 0
        fi
    done
    
    # 如果没有找到特定前缀的接口，返回第一个非无线接口
    for iface in $interfaces; do
        if [[ ! $iface == wlan* && ! $iface == wlp* ]]; then
            echo "$iface"
            return 0
        fi
    done
    
    return 1
}

# 获取有线接口
echo "正在检测有线网络接口..."
INTERFACE=$(get_wired_interface)
if [ -z "$INTERFACE" ]; then
    echo "错误：未找到有效的有线网络接口"
    exit 1
fi
echo "已检测到有线网络接口：$INTERFACE"

# 定义netplan配置文件路径
CONFIG_FILE="/etc/netplan/01-ethernet-config.yaml"

# 创建netplan配置
echo "正在创建网络配置文件..."
cat > "$CONFIG_FILE" << EOF
# 自动生成的有线网络配置：DHCP优先，失败时使用静态IP
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    $INTERFACE:
      dhcp4: true
      dhcp4-overrides:
        route-metric: 100  # DHCP路由优先级更高
      addresses: [192.168.100.100/24]  # 静态IP地址
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]  # DNS服务器
      routes:
        - to: default
          via: 192.168.100.1  # 默认网关
          metric: 200  # 静态路由优先级较低
EOF

if [ $? -ne 0 ]; then
    echo "错误：创建配置文件失败"
    exit 1
fi

# 应用netplan配置
echo "正在应用网络配置..."
if ! netplan try --timeout 10; then
    echo "错误：配置验证失败，正在回滚"
    exit 1
fi

if ! netplan apply; then
    echo "错误：应用配置失败"
    exit 1
fi

echo "网络配置成功完成！"
echo "配置详情："
echo "1. 网络接口：$INTERFACE"
echo "2. 优先使用DHCP获取IP地址"
echo "3. 当DHCP失败时，将自动使用静态IP：192.168.100.100"
exit 0
    