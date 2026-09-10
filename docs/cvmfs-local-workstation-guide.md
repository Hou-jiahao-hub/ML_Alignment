# 本地工作站 CVMFS 环境操作手册

本文说明如何在独立工作站上挂载 CERN `sft.cern.ch` CVMFS 仓库，
并加载本项目使用的 LCG 运行环境。本文不涉及项目仓库、ACTS 源码、
编译产物或实验数据的迁移。

## 1. 适用范围

推荐环境：

- x86_64 架构；
- AlmaLinux、Rocky Linux 或 RHEL 9；
- 能使用 `sudo` 和 systemd 的原生 Linux 主机或虚拟机；
- 能通过 HTTP 访问 CERN CVMFS Stratum-1 及其镜像。

项目当前加载的 LCG view 是：

```text
/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh
```

因此，其他 Linux 发行版即使能成功挂载 CVMFS，也不一定与该 EL9
二进制环境兼容。Ubuntu、WSL2 和容器的注意事项见本文附录。

> CVMFS 让工作站脱离计算集群运行，但不等于离线运行。首次访问文件时，
> 客户端仍需通过网络下载内容；断网后只能使用本机缓存中已有的内容。

## 2. 工作原理

```text
CERN Stratum-1 / 镜像
          |
          | HTTP（单机默认 DIRECT）
          v
本机 cvmfs2 + /var/lib/cvmfs 缓存
          |
          | autofs 按需挂载
          v
/cvmfs/sft.cern.ch
          |
          v
LCG_109 x86_64-el9-gcc13-opt
```

Linux 上通常由 autofs 管理 `/cvmfs`。只查看 `/cvmfs` 时可能看不到仓库；
访问 `/cvmfs/sft.cern.ch` 才会触发实际挂载。仓库空闲一段时间后，autofs
也可能自动卸载它。

## 3. 安装前检查

### 3.1 检查操作系统与架构

```bash
uname -m
cat /etc/os-release
command -v systemctl
```

预期架构为：

```text
x86_64
```

如果是 ARM/aarch64，不能直接运行本项目指定的 `x86_64` LCG view。

### 3.2 检查 FUSE

```bash
ls -l /dev/fuse
```

如果 `/dev/fuse` 不存在，先尝试：

```bash
sudo modprobe fuse
ls -l /dev/fuse
```

如果仍不存在，常见原因包括内核未启用 FUSE、虚拟机配置不完整，或者当前
shell 位于未开放 FUSE 的容器中。此时应先修复宿主机/虚拟机的 FUSE，或采用
“宿主机挂载、容器共享”的部署方式。

### 3.3 检查网络

```bash
getent hosts cvmfs-stratum-one.cern.ch
curl -fsSI --max-time 15 \
  http://cvmfs-stratum-one.cern.ch/cvmfs/sft.cern.ch/.cvmfspublished
```

常见失败含义：

- `Could not resolve host`：DNS 不可用；
- `Connection timed out`：防火墙、出口策略或代理阻止访问；
- 返回单位网关的拦截页面：HTTP 被透明代理或认证网关接管。

本手册面向少量独立工作站，使用 `DIRECT` 连接。如果同一地点有多台客户端，
应配置站点 Squid proxy，避免所有机器直接访问 Stratum-1。

## 4. AlmaLinux/RHEL/Rocky Linux 9 安装

### 4.1 添加官方软件源并安装客户端

```bash
sudo dnf install -y \
  https://cvmrepo.s3.cern.ch/cvmrepo/yum/cvmfs-release-latest.noarch.rpm

sudo dnf install -y cvmfs cvmfs-config-default autofs
```

确认安装结果：

```bash
rpm -q cvmfs cvmfs-config-default autofs
cvmfs2 --version
```

如果提示找不到软件包：

```bash
sudo dnf clean all
sudo dnf makecache
```

然后重新运行安装命令。如果添加软件源本身失败，应先检查 DNS、HTTPS 出口和
系统 CA 证书，不要从不明镜像下载 RPM。

### 4.2 初始化 CVMFS 与 autofs

```bash
sudo cvmfs_config setup
sudo systemctl enable --now autofs
```

检查 autofs 映射：

```bash
cat /etc/auto.master.d/cvmfs.autofs
```

应包含：

```text
/cvmfs /etc/auto.cvmfs
```

检查服务：

```bash
systemctl is-enabled autofs
systemctl is-active autofs
systemctl status autofs --no-pager
```

前两个命令应分别输出 `enabled` 和 `active`。

## 5. 配置单机客户端

使用本机覆盖文件，不要直接修改软件包提供的 `/etc/cvmfs/default.conf`。

如果 `/etc/cvmfs/default.local` 已存在，先查看并备份它：

```bash
sudo cat /etc/cvmfs/default.local
sudo cp -a /etc/cvmfs/default.local /etc/cvmfs/default.local.bak
```

编辑配置：

```bash
sudoedit /etc/cvmfs/default.local
```

写入或合并以下参数：

```bash
CVMFS_REPOSITORIES=sft.cern.ch
CVMFS_CLIENT_PROFILE=single
CVMFS_HTTP_PROXY=DIRECT
CVMFS_CACHE_BASE=/var/lib/cvmfs
CVMFS_QUOTA_LIMIT=4000
```

参数含义：

| 参数 | 作用 |
|---|---|
| `CVMFS_REPOSITORIES` | 只启用项目需要的 `sft.cern.ch` |
| `CVMFS_CLIENT_PROFILE` | 使用适合个人工作站的单机配置 |
| `CVMFS_HTTP_PROXY` | 不经过集群代理，直接访问镜像 |
| `CVMFS_CACHE_BASE` | 本地只读内容缓存根目录 |
| `CVMFS_QUOTA_LIMIT` | 缓存软配额，单位为 MB；此处为 4 GB |

如果磁盘空间充足，可以把缓存配额提高到 20 GB：

```bash
CVMFS_QUOTA_LIMIT=20000
```

应用配置：

```bash
sudo systemctl restart autofs
```

## 6. 配置与挂载验收

### 6.1 检查基础配置

```bash
sudo cvmfs_config chksetup
```

正常结果为：

```text
OK
```

查看 `sft.cern.ch` 的最终生效参数：

```bash
cvmfs_config showconfig -s sft.cern.ch
```

确认输出中至少包含：

```text
CVMFS_REPOSITORIES=sft.cern.ch
CVMFS_CLIENT_PROFILE=single
CVMFS_HTTP_PROXY=DIRECT
CVMFS_CACHE_BASE=/var/lib/cvmfs
```

如果最终值与 `/etc/cvmfs/default.local` 不一致，检查以下目录中是否存在优先级
更高的仓库或域级覆盖：

```bash
find /etc/cvmfs/default.d /etc/cvmfs/domain.d /etc/cvmfs/config.d \
  -maxdepth 1 -type f -print
```

### 6.2 探测并触发挂载

```bash
cvmfs_config probe sft.cern.ch
ls /cvmfs/sft.cern.ch
ls /cvmfs/sft.cern.ch/lcg
```

查看挂载状态：

```bash
findmnt /cvmfs/sft.cern.ch
cvmfs_config status sft.cern.ch
cvmfs_config stat sft.cern.ch
```

### 6.3 验证项目所需 LCG view

```bash
test -r \
  /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh \
  && echo "LCG_109 view OK"
```

如果命令没有输出，依次检查：

1. `uname -m` 是否为 `x86_64`；
2. `findmnt /cvmfs/sft.cern.ch` 是否显示 CVMFS/FUSE 挂载；
3. `cvmfs_config probe sft.cern.ch` 的错误信息；
4. LCG view 路径是否被手工改写。

## 7. 加载并验证 LCG 环境

打开一个未激活 conda 的新 shell：

```bash
bash --noprofile --norc
```

加载 LCG：

```bash
source \
  /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh
```

检查主要工具：

```bash
echo "${LCG_VERSION:-LCG_VERSION is not set}"
which python3
python3 --version
which root
root-config --version
```

本项目的完整 ACTS 环境由仓库根目录的 `setup_acts.sh` 加载。它会先检查
CVMFS，再加载上述 LCG view，并根据项目配置注入 ACTS 路径：

```bash
cd <ML_ALIGNMENT_REPOSITORY>
source setup_acts.sh
```

不要在同一个 shell 中混用 LCG 和 conda：

```text
Shell A：source setup_acts.sh，用于 ACTS simulation/reconstruction
Shell B：conda activate pytorch，用于 ML、绘图和单元测试
```

LCG 会设置 `PYTHONHOME` 等变量；在同一个 shell 中再激活 conda 可能破坏
conda Python。

## 8. 手工挂载与卸载

正常工作优先使用 autofs。只有在诊断 autofs 故障时，才临时手工挂载。

### 8.1 在独立路径诊断 CVMFS 本体

```bash
sudo mkdir -p /mnt/cvmfs-sft
sudo mount -t cvmfs sft.cern.ch /mnt/cvmfs-sft
ls /mnt/cvmfs-sft/lcg
```

检查完成后卸载：

```bash
sudo umount /mnt/cvmfs-sft
```

如果独立路径可以手工挂载，而 `/cvmfs/sft.cern.ch` 不能自动挂载，说明
CVMFS 网络和签名验证基本正常，应重点排查 autofs。

### 8.2 在标准路径手工挂载

不要让 autofs 和手工 mount 同时管理 `/cvmfs/sft.cern.ch`：

```bash
sudo systemctl stop autofs
sudo mkdir -p /cvmfs/sft.cern.ch
sudo mount -t cvmfs sft.cern.ch /cvmfs/sft.cern.ch
```

恢复 autofs：

```bash
sudo umount /cvmfs/sft.cern.ch
sudo systemctl start autofs
```

## 9. 故障排查

### 9.1 `root privileges required`

基础检查或系统初始化需要 root：

```bash
sudo cvmfs_config chksetup
sudo cvmfs_config setup
```

### 9.2 `/dev/fuse: No such file or directory`

```bash
sudo modprobe fuse
ls -l /dev/fuse
```

如果发生在容器中，应改为在宿主机挂载 CVMFS，再共享给容器；或者显式向
容器开放 `/dev/fuse` 和挂载权限。

### 9.3 `Failed to connect to bus`

当前环境没有可用的 systemd，通常表示正在精简容器或未启用 systemd 的
WSL 中。不要在这种普通容器里启动 autofs，改用宿主机挂载方案。

### 9.4 `/cvmfs` 为空

autofs 是按需挂载，直接访问完整路径：

```bash
ls /cvmfs/sft.cern.ch
```

如果仍失败：

```bash
sudo systemctl restart autofs
journalctl -u autofs -b --no-pager
```

### 9.5 下载 root catalog 失败

检查有效配置、DNS 和 HTTP：

```bash
cvmfs_config showconfig -s sft.cern.ch
getent hosts cvmfs-stratum-one.cern.ch
curl -fsSI --max-time 15 \
  http://cvmfs-stratum-one.cern.ch/cvmfs/sft.cern.ch/.cvmfspublished
```

如果单位网络要求代理，把 `CVMFS_HTTP_PROXY=DIRECT` 替换为管理员提供的
代理地址，并重启 autofs。不要把普通 shell 的 `HTTP_PROXY` 当作 CVMFS
代理配置；CVMFS 使用自己的 `CVMFS_HTTP_PROXY` 参数。

### 9.6 签名或 public key 错误

检查 CERN 配置和密钥包：

```bash
rpm -q cvmfs-config-default
find /etc/cvmfs/keys/cern.ch -maxdepth 1 -type f -print
```

必要时重新安装官方配置包：

```bash
sudo dnf reinstall -y cvmfs-config-default
sudo systemctl restart autofs
```

### 9.7 `Transport endpoint is not connected`

先查找仍在占用仓库的进程：

```bash
cvmfs_config fuser sft.cern.ch
```

正常结束相关程序，然后重启 autofs：

```bash
sudo systemctl restart autofs
```

不要在不确认占用进程和挂载目标的情况下强制卸载或杀死所有 CVMFS 进程。

### 9.8 缓存空间不足

```bash
df -h /var/lib/cvmfs
cvmfs_config stat sft.cern.ch
```

增大 `/etc/cvmfs/default.local` 中的 `CVMFS_QUOTA_LIMIT` 后：

```bash
sudo systemctl restart autofs
```

`cvmfs_config wipecache` 会删除本机下载缓存，不能作为常规修复命令。只有在
确认缓存损坏、仓库均已卸载且能够重新下载内容时才应使用。

### 9.9 LCG 加载后出现 `GLIBC` 或 `GLIBCXX` 错误

这通常是宿主发行版与 `x86_64-el9-gcc13-opt` view 不兼容。不要通过替换
系统 glibc 修复；应改用 AlmaLinux/Rocky/RHEL 9 主机、虚拟机或兼容容器。

## 10. Ubuntu/Debian 附录

安装 CVMFS：

```bash
wget https://cvmrepo.s3.cern.ch/cvmrepo/apt/cvmfs-release-latest_all.deb
sudo dpkg -i cvmfs-release-latest_all.deb
rm cvmfs-release-latest_all.deb

sudo apt-get update
sudo apt-get install -y cvmfs cvmfs-config-default autofs
sudo cvmfs_config setup
sudo systemctl enable --now autofs
```

后续 `/etc/cvmfs/default.local` 配置和验收步骤与前文一致。但 Ubuntu 只保证
CVMFS 客户端可以挂载，不保证 EL9 LCG view 中所有二进制兼容。

## 11. WSL2 附录

按照对应 Linux 发行版安装和配置 CVMFS。每次启动 WSL2 发行版后运行：

```bash
sudo cvmfs_config wsl2_start
```

然后执行：

```bash
cvmfs_config probe sft.cern.ch
ls /cvmfs/sft.cern.ch/lcg
```

如果需要长期稳定运行 ACTS/LCG，优先考虑 AlmaLinux 9 虚拟机，而不是依赖
WSL2 的 FUSE 与 systemd 行为。

## 12. 容器附录

### 12.1 推荐：宿主机挂载后共享

先在宿主机触发挂载：

```bash
ls /cvmfs/sft.cern.ch/lcg
```

将整个 autofs 管理区域以 shared propagation 传入容器：

```bash
docker run -it --rm \
  -v /cvmfs:/cvmfs:shared \
  almalinux:9
```

该方式让多个容器共享宿主机 CVMFS 缓存，也避免在普通工作容器中运行
autofs 和高权限 FUSE mount。

### 12.2 CVMFS service container

如果宿主机适合通过服务容器统一提供 `/cvmfs`：

```bash
docker pull registry.cern.ch/cvmfs/service:latest

docker run -d --rm \
  --name cvmfs-service \
  -e CVMFS_CLIENT_PROFILE=single \
  -e CVMFS_REPOSITORIES=sft.cern.ch \
  -e CVMFS_HTTP_PROXY=DIRECT \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  -v /cvmfs:/cvmfs:shared \
  registry.cern.ch/cvmfs/service:latest
```

宿主机仍必须提供 `/dev/fuse`，并允许 shared mount propagation。停止该容器
会同时停止它提供的 CVMFS 挂载：

```bash
docker stop cvmfs-service
```

## 13. 最终验收清单

依次执行：

```bash
uname -m
ls -l /dev/fuse
sudo cvmfs_config chksetup
cvmfs_config showconfig -s sft.cern.ch
cvmfs_config probe sft.cern.ch
ls /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh
findmnt /cvmfs/sft.cern.ch
```

然后在新的非 conda shell 中执行：

```bash
source \
  /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh

python3 --version
root-config --version
```

满足以下条件即可认为 CVMFS/LCG 环境部署完成：

- 架构为 `x86_64`；
- `/dev/fuse` 存在；
- `cvmfs_config chksetup` 输出 `OK`；
- `cvmfs_config probe sft.cern.ch` 成功；
- `findmnt` 显示 `/cvmfs/sft.cern.ch` 已挂载；
- LCG_109 setup 脚本可读；
- 加载 LCG 后 Python 和 ROOT 可执行。

## 14. 官方参考

- [CernVM-FS Getting Started](https://cvmfs.readthedocs.io/en/2.13/cpt-quickstart.html)
- [CernVM-FS Client Configuration](https://cvmfs.readthedocs.io/en/stable/cpt-configure/)
- [Container Images and CernVM-FS](https://cvmfs.readthedocs.io/en/stable/cpt-containers.html)
- [CernVM-FS Release Notes](https://cvmfs.readthedocs.io/en/stable/cpt-releasenotes.html)
