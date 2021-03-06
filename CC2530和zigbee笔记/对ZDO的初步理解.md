---
title: 对ZDO的初步理解
categories: CC2530和zigbee笔记
date: 2019-02-05 13:01:29
---
&emsp;&emsp;`ZDO`其实是`ZigBee`协议栈中的一个协议，负责所有设备的管理和安全方案。`ZDO`就好像是一个驻留在所有`ZigBee`节点中特殊应用对象，是应用层其他端点与应用子层管理实体交互的中间件。`ZDO`占用每个节点(`node`)的端口`0`(`Endpoint0`)。<!--more-->
&emsp;&emsp;`ZDO`的配置叫做`ZDP`(`ZigBee Device Profile`，`ZigBee`设备配置)，`ZDP`可以被应用终端(`application end points`)和`ZigBee`节点访问。
&emsp;&emsp;`ZDO`是一个特殊的应用层的端口(`Endpoint`)，它是应用层其他端点与应用子层管理实体交互的中间件。它主要提供的功能如下：

- 初始化应用支持子层、网络层。
- 发现节点和节点功能。在无信标的网络中，加入的节点只对其父节点可见。而其他节点可以通过`ZDO`的功能来确定网络的整体拓扑结构，以及节点所能提供的功能。
- 安全加密管理：主要包括安全`key`的建立和发送，以及安全授权。
- 网络的维护功能。
- 绑定管理：绑定的功能由应用支持子层提供，但是绑定功能的管理却是由`ZDO`提供，它确定了绑定表的大小、绑定的发起和绑定的解除等功能。
- 节点管理：对于网络协调器和路由器，`ZDO`提供网络监测、获取路由和绑定信息、发起脱离网络过程等一系列节点管理功能。

&emsp;&emsp;`ZDO`实际上是介于应用层端点和应用支持子层中间的端点，其主要功能集中在网络管理和维护上。应用层的端点可以通过`ZDO`提供的功能来获取网络或是其他节点的信息，包括网络的拓扑结构、其它节点的网络地址和状态以及其他节点的类型和提供的服务等信息。