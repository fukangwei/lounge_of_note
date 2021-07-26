---
title: 协议栈API函数使用说明
categories: CC2530和zigbee笔记
date: 2019-10-07 15:09:47
---
# 介绍

&emsp;&emsp;这份文档为目前发布的`ZigBee 2007`协议栈提供了应用程序接口函数的使用说明。为使得我们更好地开发和理解`ZigBee`项目，这份文档详细的讲述了协议栈中的数据结构和函数调用。首先来了解一下在`ZigBee 2007`协议栈中使用的各个层次：<!--more-->

- `ZDO`：设备对象层，即`ZDO`(`ZigBee Device Object`)层，提供了管理一个`ZigBee`节点所要使用的功能函数。`ZDO`的`API`为协调器、路由器和终端设备提供了应用端点的管理函数，其中包括建立、发现和加入一个`ZigBee`网络，绑定应用端点和安全管理。
- `AF`：应用框架层，即`AF`(`Application Framework`)，提供了针对协议栈的应用端点(`EndPoint1`至`240`)和设备对象端点(`EndPoint0`)接口，其中主要包含设备描述数据结构和数据收、发函数。
- `APS`：应用支持子层，即`APS`(`Application Support Sublayer`)，为设备对象和应用实体提供了一系列的支持服务。
- `NWK`：网络层，即`NWK`(`ZigBee network`)，为上层提供了管理服务和数据服务。
- `ZMAC`：介质访问层，即`ZMAC`，在`802.15.4 MAC`与网络层之间提供接口。

# 应用函数接口

&emsp;&emsp;接下来我们讲述的应用函数接口主要包含一些经常使用的数据结构和各层提供的关键性函数。

## 设备对象(ZDO)

&emsp;&emsp;本节列举出了在`ZigBee`设备规范，即`ZigBee Device Profile`(`ZDP`)，所定义的相关命令和响应函数。

### 概述

&emsp;&emsp;`ZDP`描述了`ZDO`内部一般性的`ZigBee`设备功能是如何实现的。它定义了使用命令和响应对的设备描述和簇。`ZDP`为`ZDO`和应用程序提供如下功能：设备网络启动、设备和服务发现、终端设备绑定、绑定和取消绑定服务、网络管理服务。

### ZDO网络设备启动

&emsp;&emsp;通过默认的`ZDApp_Init`函数(在`ZDApp.c`中)启动`ZigBee`网络中的设备。为了使应用接管网络的启动，必须在编译选项中包含`HOLD_AUTO_START`，推荐同时包含`NV_RESTORE`(储存`ZigBee`网络状态到`NV`)编译选项，包含这个两个编译选项后就需要调用`ZDOInitDevice`来启动网络中的设备。但是一个应用程序可以跳过这个默认行为(`ZDApp.c`)：

``` cpp
#if defined( HOLD_AUTO_START )
    devStates_t devState = DEV_HOLD;
#else
    devStates_t devState = DEV_INIT;
#endif
```

其中`HOLD_AUTO_START`在`IAR -> Project -> Option -> C/C++ Compile -> Preprocess -> Defined symbols`中定义。如果预编译选择`HOLD_AUTO_START`，则`ZDO`不启动网络设备，只是闪烁`LED4`，交由应用程序启动网络设备(`ZDApp.c`)：

``` cpp
void ZDApp_Init ( uint8 task_id ) {
    if ( devState != DEV_HOLD ) {
        ZDOInitDevice ( 0 );
    } else {
        /* Blink LED to indicate HOLD_START */
        HalLedBlink ( HAL_LED_4, 0, 50, 500 );
    }
}
```

在`sapi.c`文件中有如下内容：

``` cpp
UINT16 SAPI_ProcessEvent ( byte task_id, UINT16 events ) {
    if ( events & ZB_ENTRY_EVENT ) {
        if ( startOptions & ZCD_STARTOPT_AUTO_START ) {
            zb_StartRequest(); /* 即调用ZDOInitDevice (zgStartDelay) */
        } else {
            /* blink leds and wait for external input to config and restart */
            HalLedBlink ( HAL_LED_2, 0, 50, 500 );
        }
    }
}
```

`ZDOInitDevice`函数原型如下：

``` cpp
uint8 ZDOInitDevice ( uint16 startDelay );
```

参数`startDelay`为设备启动延时(毫秒)，返回值如下：

返回值                                | 说明
-------------------------------------|-----
`ZDO_INITDEV_RESTORED_NETWORK_STATE` | 网络状态为恢复
`ZDO_INITDEV_NEW_NETWORK_STATE`      | 网络状态为初始化
`ZDO_INITDEV_LEAVE_NOT_STARTED`      | 网络状态为未启动

&emsp;&emsp;如果应用想要强制一个`新的`加入(即不恢复网络状态)，则应该先调用：

``` cpp
zgWriteStartupOptions ( ZG_STARTUP_SET, ZCD_STARTOPT_DEFAULT_NETWORK_STATE );
```

来设置`ZCD_NV_STARTUP_OPTION`中的`ZCD_STARTOPT_DEFAULT_NETWORK_STATE`位。

### ZDO信息回调函数

&emsp;&emsp;通过函数`ZDO_RegisterForZDOMsg`注册请求或响应消息，这样就可以接收其他设备无线传输的消息。

#### ZDO_RegisterForZDOMsg

&emsp;&emsp;调用该函数可以将接收到的无线传输信息复制一份到`OSAL`层的某个任务。该任务接收到消息后，可以自行解析此消息或者通过`ZDO`解析函数解析此消息，只有响应消息需要使用`ZDO`解析函数。

``` cpp
ZStatus_t ZDO_RegisterForZDOMsg ( uint8 taskID, uint16 clusterID );
```

- `taskID`：任务`ID`，该任务发送`OSAL`消息。
- `clustered`：簇`ID`(例如`NWK_addr_rsp`)，在`ZDProfile.h`定义了相关的簇。

返回值`ZStatus_t`为状态。

&emsp;&emsp;消息接收到之后作为`ZDO_CB_Msg`(一种系统消息)发送至应用或者任务。消息结构体`zdoIncomingMsg_t`定义在`ZDProfile.h`中：

``` cpp
#define ZDO_CB_MSG 0xD3 /* ZDO incoming message callback */

typedef struct {
    osal_event_hdr_t hdr;
    zAddrType_t srcAddr;
    uint8 wasBroadcast;
    cId_t clusterID;
    uint8 SecurityUse;
    uint8 TransSeq;
    uint8 asduLen;
    uint16 macDestAddr;
    uint8 *asdu;
} zdoIncomingMsg_t;
```

在应用层使用该函数注册的消息有：

``` cpp
void SAPI_Init ( byte task_id ) {
    /* Register callback evetns from the ZDApp */
    ZDO_RegisterForZDOMsg ( sapi_TaskID, NWK_addr_rsp );
    ZDO_RegisterForZDOMsg ( sapi_TaskID, Match_Desc_rsp );
}
```

调用`ZDO_RegisterForZDOMsg`在应用层注册后，应用层处理接收到的消息方式如下：

``` cpp
UINT16 SAPI_ProcessEvent ( byte task_id, UINT16 events ) {
    osal_event_hdr_t *pMsg;

    if ( events & SYS_EVENT_MSG ) {
        pMsg = ( osal_event_hdr_t * ) osal_msg_receive ( task_id );

        while ( pMsg ) {
            switch ( pMsg->event ) {
                case ZDO_CB_MSG:
                    SAPI_ProcessZDOMsgs ( ( zdoIncomingMsg_t * ) pMsg );
                    break;
            }
        }
    }
}
```

在`ZDO`层使用该函数注册的消息有：

``` cpp
void ZDApp_RegisterCBs ( void ) {
#if defined ( ZDO_IEEEADDR_REQUEST ) || defined ( REFLECTOR )
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, IEEE_addr_rsp );
#endif

#if defined ( ZDO_NWKADDR_REQUEST ) || defined ( REFLECTOR )
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, NWK_addr_rsp );
#endif

#if ZG_BUILD_COORDINATOR_TYPE
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, Bind_rsp );
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, Unbind_rsp );
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, End_Device_Bind_req );
#endif

#if defined ( REFLECTOR )
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, Bind_req );
    ZDO_RegisterForZDOMsg ( ZDAppTaskID, Unbind_req );
#endif
}
```

调用`ZDO_RegisterForZDOMsg`在`ZDO`层注册后，`ZDO`层处理接收到的消息方式如下：

``` cpp
void ZDApp_ProcessOSALMsg ( osal_event_hdr_t *msgPtr ) {
    switch ( msgPtr->event ) {
        case ZDO_CB_MSG:
            ZDApp_ProcessMsgCBs ( ( zdoIncomingMsg_t * ) msgPtr );
            break;
    }
}
```

#### ZDO_RemoveRegistedCB

&emsp;&emsp;调用此函数取消请求无线传输的消息。

``` cpp
ZStatus_t ZDO_RemoveRegistedCB ( uint8 taskID, uint16 clusterID );
```

- `taskID`：任务`ID`，该任务`ID`必须与`ZDO_RegisterForZDOMsg`所注册的任务`ID`相同。
- `clustered`：簇`ID`，该簇`ID`必须与`ZDO_RegisterForZDOMsg`所使用的簇`ID`相同。

返回值`ZStatus_t`为状态。

### ZDO发现API

&emsp;&emsp;`ZDO`发现`API`包含建立和发送`ZDO`设备和服务发现请求和响应。所有这些`API`函数和`ZDP`命令(`ZigBee Device Profile Command`)如下表：

API函数               | ZDP命令
---------------------|-------
`ZDP_NwkAddrReq`     | `NWK_addr_req`
`ZDP_NWKAddrRsp`     | `NWK_addr_rsp`
`ZDP_IEEEAddrReq`    | `IEEE_addr_req`
`ZDP_IEEEAddrRsp`    | `IEEE_addr_rsp`
`ZDP_NodeDescReq`    | `Node_Desc_req`
`ZDP_NodeDescRsp`    | `Node_Desc_rsp`
`ZDP_PowerDescReq`   | `Power_Desc_req`
`ZDP_PowerDescRsp`   | `Power_Desc_rsp`
`ZDP_SimpleDescReq`  | `Simple_Desc_req`
`ZDP_SimpleDescRsp`  | `Simple_Desc_rsp`
`ZDP_ComplexDescReq` | `Complex_Desc_req`
`ZDP_ActiveEPIFReq`  | `Active_EP_req`
`ZDP_ActiveEPIFRsp`  | `Active_EP_rsp`
`ZDP_MatchDescReq`   | `Match_Desc_req`
`ZDP_MatchDescRsp`   | `Match_Desc_rsp`
`ZDP_UserDescSet`    | `User_Desc_set`
`ZDP_UserDescConf`   | `User_Desc_conf`
`ZDP_UserDescReq`    | `User_Desc_req`
`ZDP_UserDescRsp`    | `User_Desc_rsp`
`ZDP_DeviceAnnce`    | `Device_annce`
`ZDP_ServerDiscReq`  | `System_Server_Discovery_req`
`ZDP_ServerDiscRsp`  | `System_Server_Discovery_rsp`

#### ZDP_NwkAddrReq

&emsp;&emsp;调用此函数将生成一个根据已知`IEEE`地址询问远程节点`16`位网络地址的消息，这个消息作为一个广播消息发送至网络中的所有节点。

``` cpp
afStatus_t ZDP_NwkAddrReq ( byte *IEEEAddress, byte ReqType, byte StartIndex, byte SecuritySuite );
```

- `IEEEAddress`：远程节点的`IEEE`地址。
- `ReqType`：有如下选项：

选项                           | 说明
-------------------------------|-----
`ZDP_NWKADDR_REQTYPE_SINGLE`   | 只返回节点的短地址和扩展地址
`ZDP_NWKADDR_REQTYPE_EXTENDED` | 返回节点的短地址和扩展地址以及所有相关节点的短地址

- `StartIndex`：响应节点的响应信息可以有很多的响应选项，请求程序可以指定一个起始索引号，索引从`0`开始。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_NWKAddrRsp

&emsp;&emsp;`ZDP_NWKAddrRsp`实际上是调用`ZDP_ADDRRsp`这个宏定义，用于建立和发送`16`位网络地址响应。

``` cpp
afStatus_t ZDP_NWKAddrRsp (
    byte TranSeq,
    zAddrType_t *dstAddr,
    byte Status,
    byte *IEEEAddrRemoteDev,
    byte ReqType,
    uint16 nwkAddr,
    byte NumAssocDev,
    byte StartIndex,
    uint16 *NWKAddrAssocDevList,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_INVALID_REQTYPE`  | `1`
`ZDP_DEVICE_NOT_FOUND` | `2`

- `IEEEAddrRemoteDev`：远程节点的`64`位`IEEE`地址。
- `ReqType`：请求的类型。
- `nwkAddr`：远程节点的`16`位网络地址。
- `NumAssocDev`：与远程节点关联的节点数目。
- `StartIndex`：响应节点的响应信息可以有很多的响应选项，请求程序可以指定一个起始索引号，该索引号是响应信息的起始索引号。
- `NWKAddrAssocDevList`：与远程节点关联的节点`16`位网络地址列表。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 网络地址请求和响应应用举例分析

&emsp;&emsp;本地节点首先需要在应用层或`ZDO`层注册响应信息，本例中在`sapi.c`中注册了`NWK_addr_rsp`信息，然后调用`ZDP_NwkAddrReq`请求远程节点的网络地址。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，则根据`Cluster ID`选择处理函数，本例中使用`zdpProcessAddrReq`来处理网络地址请求。当处理完之后，通过调用`fillAndSend`将响应信息发送至本地节点。由于在应用层中注册了该响应信息，因此调用`SAPI_ProcessZDOMsgs`来处理响应信息。

#### ZDP_IEEEAddrReq

&emsp;&emsp;调用此函数将生成一个根据已知`16`位网络地址询问远程节点`64`位`IEEE`地址的消息，这个消息直接单播发送至该远程节点。

``` cpp
afStatus_t ZDP_IEEEAddrReq ( uint16 shortAddr, byte ReqType, byte StartIndex, byte SecuritySuite );
```

- `shortAddr`：远程节点的`16`位网络地址。
- `ReqType`：有如下选项：

选项                           | 说明
-------------------------------|------
`ZDP_NWKADDR_REQTYPE_SINGLE`   | 只返回节点的短地址和扩展地址
`ZDP_NWKADDR_REQTYPE_EXTENDED` | 返回节点的短地址和扩展地址以及所有相关节点的短地址

- `StartIndex`：响应节点的响应信息可以有很多的响应选项，请求程序可以指定一个起始索引号，索引从`0`开始。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_IEEEAddrRsp

&emsp;&emsp;`ZDP_IEEEAddrRsp`实际上是调用`ZDP_ADDRRsp`这个宏定义，用于建立和发送`64`位`IEEE`地址响应。

``` cpp
afStatus_t ZDP_IEEEAddrRsp (
    byte TranSeq,
    zAddrType_t *dstAddr,
    byte Status,
    byte *IEEEAddrRemoteDev,
    byte ReqType,
    uint16 nwkAddr,
    byte NumAssocDev,
    byte StartIndex,
    uint16 *NWKAddrAssocDevList,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_INVALID_REQTYPE`  | `1`
`ZDP_DEVICE_NOT_FOUND` | `2`

- `IEEEAddrRemoteDev`：远程节点的`64`位`IEEE`地址。
- `ReqType`：请求的类型。
- `nwkAddr`：远程节点的`16`位网络地址。
- `NumAssocDev`：与远程节点关联的节点数目。
- `StartIndex`：响应节点的响应信息可以有很多的响应选项，请求程序可以指定一个起始索引号，该索引号是响应信息的起始索引号。
- `NWKAddrAssocDevList`：与远程节点关联的节点`16`位网络地址列表。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### IEEE地址请求和响应应用举例分析

&emsp;&emsp;本地节点首先需要在应用层或`ZDO`层注册响应信息，本例在`ZDApp.c`中注册了`IEEE_addr_rsp`信息，然后调用`ZDP_IEEEAddrReq`请求远程节点的网络地址。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，本例使用`zdpProcessAddrReq`来处理`IEEE`地址请求。当处理完之后，通过调用`fillAndSend`将响应信息发送至本地节点。由于在`ZDO`中注册了该响应信息，因此调用`ZDApp_ProcessMsgCBs`来处理响应信息。

#### ZDP_NodeDescReq

&emsp;&emsp;`ZDP_NodeDescReq`实际上是调用宏定义`ZDP_NWKAddrOfInterestReq`，这个函数建立和发送一个节点描述符(`Node Descriptor`)请求至已明确网络地址的远程节点。

``` cpp
afStatus_t ZDP_NodeDescReq ( zAddrType_t *dstAddr, uint16 NWKAddrOfInterest, byte SecuritySuite );
```

- `DstAddr`：目的地址。
- `NWKAddrOfInterest`：远程节点的`16`位网络地址。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_NodeDescMsg

&emsp;&emsp;调用此函数响应节点描述符的请求。

``` cpp
afStatus_t ZDP_NodeDescMsg (
    byte TransSeq,
    zAddrType_t *dstAddr,
    byte Status,
    uint16 nwkAddr,
    NodeDescriptorFormat_t *pNodeDesc,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_DEVICE_NOT_FOUND` | `1`

- `nwkAddr`：已明确的远程节点的`16`位网络地址。
- `pNodeDesc`：节点描述符。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 节点描述符请求和响应应用举例分析

&emsp;&emsp;节点描述符包含`ZigBee`节点的功能信息，每个节点只有唯一的一个节点描述符，其数据结构如下(`AF.h`)：

``` cpp
typedef struct {
    uint8 LogicalType: 3;        /* 节点逻辑类型。000：协调器；001：路由器；010：终端设备           */
    uint8 ComplexDescAvail: 1;   /* 可否使用复杂描述符。1：可以；0：不可以                         */
    uint8 UserDescAvail: 1;      /* 可否使用用户描述符。1：可以；0：不可以                         */
    uint8 Reserved: 3;           /* 保留位                                                      */
    uint8 APSFlags: 3;           /* 应用支持子层功能标志位。一般为0，表示不使用                    */
    uint8 FrequencyBand: 5;      /* 频段位。0：868至868.6MHz；2：902至928MHz；3：2400至2483.5MHz  */
    uint8 CapabilityFlags;       /* 功能标志位。备用协调器、设备类型、电源来源、RXON、安全、分配地址 */
    uint8 ManufacturerCode[2];   /* 制造商代码。由ZigBee联盟分配                                  */
    uint8 MaxBufferSize;         /* 最大缓冲区。网络层数据单元(NSDU)的最大值，以字节为单位          */
    uint8 MaxInTransferSize[2];  /* 最大输入。应用支持子层数据单元(ASDU)的最大值，以字节为单位      */
    uint16 ServerMask;           /* 服务器掩码。第0至6位可用，其他位保留                          */
    uint8 MaxOutTransferSize[2]; /* 最大输出。应用支持子层数据单元(ASDU)的最大值，以字节为单位      */
    uint8 DescriptorCapability;  /* 描述符能力域。第0至1位可用，其他位保留                        */
} NodeDescriptorFormat_t;
```

本地节点在`ZDApp.c`中调用`ZDConfig_UpdateNodeDescriptor`初始化节点描述符，本例在应用层或`ZDO`层均未注册`Node_Desc_rsp`信息。调用`ZDP_NodeDescReq`可以根据已明确了的网络地址请求远程节点的节点描述符。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，会使用`zdpProcessNodeDescReq`来处理节点描述符请求。当处理完之后，通过调用`ZDP_NodeDescMsg`将响应信息发送至本地节点。

#### 2.1.4.10 ZDP_PowerDescReq

&emsp;&emsp;`ZDP_PowerDescReq`实际上是调用宏定义`ZDP_NWKAddrOfInterestReq`。这个函数建立和发送一个电源描述符(`Power Descriptor`)请求至已明确网络地址的远程节点。

``` cpp
afStatus_t ZDP_PowerDescReq ( zAddrType_t *dstAddr, uint16 NWKAddrOfInterest, byte SecuritySuite );
```

- `DstAddr`：目的地址。
- `NWKAddrOfInterest`：远程节点的`16`位网络地址。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_PowerDescMsg

&emsp;&emsp;调用此函数响应电源描述符的请求。

``` cpp
afStatus_t ZDP_PowerDescMsg (
    byte TransSeq,
    zAddrType_t *dstAddr,
    byte Status,
    uint16 nwkAddr,
    NodePowerDescriptorFormat_t *pPowerDesc,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_DEVICE_NOT_FOUND` | `1`

- `nwkAddr`：已明确的远程节点的`16`位网络地址。
- `pPowerDesc`：电源描述符。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 电源描述符请求和响应应用举例分析

&emsp;&emsp;电源描述符动态指示`ZigBee`节点的电源状态信息，每个节点只有唯一的一个电源描述符，其数据结构如下(`AF.h`)：

``` cpp
typedef struct {
    unsigned int PowerMode: 4;               /* 电源模式          */
    unsigned int AvailablePowerSources: 4;   /* 可用电源来源      */
    unsigned int CurrentPowerSource: 4;      /* 当前电源来源      */
    unsigned int CurrentPowerSourceLevel: 4; /* 当前电源来源的电量 */
} NodePowerDescriptorFormat_t;
```

本地节点在`ZDApp.c`中调用`ZDConfig_UpdatePowerDescriptor`初始化电源描述符，本例应用层或`ZDO`层均未注册`Power_Desc_rsp`信息。调用`ZDP_PowerDescReq`可以根据已明确了的网络地址请求远程节点的节点描述符。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，会使用`zdpProcessPowerDescReq`来处理电源描述符请求。当处理完之后，通过调用`ZDP_PowerDescMsg`将响应信息发送至本地节点。

#### ZDP_SimpleDescReq

&emsp;&emsp;`ZDP_SimpleDescReq`函数建立和发送一个简单描述符(`Simple Descriptor`)请求至已明确网络地址的远程节点。

``` cpp
afStatus_t ZDP_SimpleDescReq ( zAddrType_t *dstAddr, uint16 nwkAddr, byte endPoint, byte SecurityEnable )
```

- `dstAddr`：目的地址。
- `NWKAddr`：远程节点的`16`位网络地址。
- `Endpoint`：端点号。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_SimpleDescMsg

&emsp;&emsp;调用此函数响应简单描述符的请求。

``` cpp
afStatus_t ZDP_SimpleDescMsg (
    byte TransSeq,
    zAddrType_t *dstAddr,
    byte Status,
    SimpleDescriptorFormat_t *pSimpleDesc,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态               | 数值
-------------------|-----
`UCCESS`           | `0`
`INVALID_EP`       | `1`
`NOT_ACTIVE`       | `2`
`DEVICE_NOT_FOUND` | `3`

- `pSimpleDesc`：简单描述符。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 简单描述符请求和响应应用举例分析

&emsp;&emsp;简单描述符包含该节点所含每个端点的具体信息，其数据结构如下(`AF.h`)：

``` cpp
typedef struct {
    byte EndPoint;             /* 端点          */
    uint16 AppProfId;          /* 应用规范标识符 */
    uint16 AppDeviceId;        /* 应用设备标识符 */
    byte AppDevVer: 4;         /* 应用设备版本   */
    byte Reserved: 4;          /* 保留          */
    byte AppNumInClusters;     /* 应用输入簇个数 */
    cId_t *pAppInClusterList;  /* 应用输入簇列表 */
    byte AppNumOutClusters;    /* 应用输出簇个数 */
    cId_t *pAppOutClusterList; /* 应用输出簇列表 */
} SimpleDescriptionFormat_t;
```

本例应用层或`ZDO`层均未注册`Simple_Desc_rsp`信息。调用`ZDP_SimpleDescReq`可以根据已明确了的网络地址请求远程节点的某个指定端点的简单描述符。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，会使用`zdpProcessSimpleDescReq`来处理简单描述符请求。当处理完之后，通过调用`ZDP_SimpleDescMsg`将响应信息发送至本地节点。

#### ZDP_ActiveEPIFReq

&emsp;&emsp;`ZDP_ActiveEPIFReq`实际上是调用宏定义`ZDP_NWKAddrOfInterestReq`。这个函数建立和发送一个活动端点请求至已明确网络地址的远程节点。使用这个宏定义请求远程节点所有活动的端点。

``` cpp
afStatus_t ZDP_ActiveEPIFReq ( zAddrType_t *dstAddr, uint16 NWKAddrOfInterest, byte SecuritySuite );
```

- `DstAddr`：目的地址。
- `NWKAddrOfInterest`：远程节点的`16`位网络地址。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_ActiveEPIFRsp

&emsp;&emsp;该函数实际上是调用宏定义`ZDP_EPIFRsp`，调用此函数响应活动端点请求。

``` cpp
afStatus_t ZDP_ActiveEPIFRsp (
    byte TranSeq,
    zAddrType_t *dstAddr,
    byte Status,
    uint16 nwkAddr,
    byte Count,
    byte *pEPIntfList,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_DEVICE_NOT_FOUND` | `1`

- `nwkAddr`：已明确的远程节点的`16`位网络地址。
- `Count`：活动端点个数。
- `pEPIntfList`：活动端点列表。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 活动端点请求和响应应用举例分析

&emsp;&emsp;本例应用层或`ZDO`层均未注册`Active_EP_rsp`信息。调用`ZDP_ActiveEPIFReq`可以根据已明确了的网络地址请求远程节点的活动端点。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，会使用`zdpProcessActiveEPReq`来处理活动端点请求。当处理完之后，通过调用如下函数：

``` cpp
ZDP_ActiveEPRsp (
    inMsg->TransSeq, & ( inMsg->srcAddr ), stat, aoi,
    cnt, ( uint8 * ) ZDOBuildBuf, inMsg->SecurityUse );
```

将响应信息发送至本地节点。

#### ZDP_MatchDescReq

&emsp;&emsp;这个函数将建立和发送一个匹配描述符请求。使用这个函数查询与本地节点的输入或输出簇列表相匹配的远程节点。

``` cpp
afStatus_t ZDP_MatchDescReq (
    zAddrType_t *dstAddr,
    uint16 nwkAddr,
    uint16 ProfileID,
    byte NumInClusters,
    byte *InClusterList,
    byte NumOutClusters,
    byte *OutClusterList,
    byte SecuritySuite
);
```

- `DstAddr`：目的地址。
- `nwkAddr`：已明确的`16`位网络地址。
- `ProfileID`：应用规范`ID`号。
- `NumInClusters`：输入簇的个数。
- `InClusterList`：输入簇列表，每个元素`1 byte`。
- `NumOutClusters`：输出簇的个数。
- `OutClusterList`：输出簇列表，每个元素`1 byte`。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_MatchDescRsp

&emsp;&emsp;该函数实际上是调用宏定义`ZDP_EPIFRsp`，调用此函数响应匹配描述符请求。

``` cpp
afStatus_t ZDP_MatchDescRsp (
    byte TranSeq,
    zAddrType_t *dstAddr,
    byte Status,
    uint16 nwkAddr,
    byte Count,
    byte *pEPIntfList,
    byte SecuritySuite
);
```

- `TranSeq`：传输序号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态                   | 数值
-----------------------|-----
`ZDP_SUCCESS`          | `0`
`ZDP_DEVICE_NOT_FOUND` | `1`

- `nwkAddr`：已明确的远程节点的`16`位网络地址。
- `Count`：活动端点个数。
- `pEPIntfList`：活动端点列表。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 匹配描述符请求和响应应用举例分析

&emsp;&emsp;本地节点首先需要在应用层或`ZDO`层注册响应信息，本例在`sapi.c`中注册了`Match_Desc_rsp`信息。另外，远程节点需要开启匹配描述符响应。然后调用`ZDP_MatchDescReq`请求远程节点的匹配描述符。远程节点接收到该请求信息(该信息从属于`AF_DATA_CONFIRM_CMD`)，根据`Cluster ID`选择处理函数，本例中使用`ZDO_ProcessMatchDescReq`来处理匹配描述符请求。当处理完之后，通过调用`fillAndSend`将响应信息发送至本地节点。由于在应用层中注册了该响应信息，因此调用`SAPI_ProcessZDOMsgs`来处理响应信息。最后通过调用应用支持子层的函数`APSME_BindReques`建立绑定表。

#### ZDP_DeviceAnnce

&emsp;&emsp;这个函数为`ZigBee`终端设备建立和发送一个`End_Device_annce`命令，来通知网络中的其他`ZigBee`节点。该命令包含终端设备的`16`位网络地址和`64`位`IEEE`地址以及容量。接收到`End_Device_annce`命令的节点将检查`IEEE`地址并更新相应的网络地址，但不会对此命令返回响应信息。

``` cpp
afStatus_t ZDP_DeviceAnnce ( uint16 nwkAddr, byte *IEEEAddr, byte capabilities, byte SecurityEnable );
```

- `nwkAddr`：本地节点的`16`位网络地址。
- `IEEEAddr`：本地节点的`64`位`IEEE`地址。
- `Capabilities`：本地节点的容量。
- `SecurityEnable`：安全要求。

返回值`ZStatus_t`为状态。

### ZDO绑定API

&emsp;&emsp;`ZDO`绑定`API`建立和发送`ZDO`绑定请求和响应。所有的绑定表建立在`ZigBee`协调器中。因此，只有`ZigBee`协调器可以接收绑定请求。

ZDO绑定API             | ZDP绑定服务命令
-----------------------|---------------
`ZDP_EndDeviceBindReq` | `End_Device_Bind_req`
`ZDP_EndDeviceBindRsp` | `End_Device_Bind_rsp`
`ZDP_BindReq`          | `Bind_req`
`ZDP_BindRsp`          | `Bind_rsp`
`ZDP_UnbindReq`        | `Unbind_req`
`ZDP_UnbindRsp`        | `Unbind_rsp`

#### ZDP_EndDeviceBindReq

&emsp;&emsp;调用这个函数将建立和发送一个终端设备绑定请求(手动绑定)。在手动绑定建立之后，我们可以间接发送信息(无地址)至协调器，协调器将发送此消息至绑定的节点。通过同样的方式，我们也可以接收到绑定节点的信息。

``` cpp
afStatus_t ZDP_EndDeviceBindReq (
    zAddrType_t *dstAddr,
    uint16 LocalCoordinator,
    byte ep,
    uint16 ProfileID,
    byte NumInClusters,
    byte *InClusterList,
    byte NumOutClusters,
    byte *OutClusterList,
    byte SecuritySuite
);
```

- `DstAddr`：目的地址。
- `LocalCoordinator`：已知协调器的`16`位网络地址。
- `ep`：端点。
- `ProfileID`：应用规范`ID`号。
- `NumInClusters`：输入簇的个数。
- `InClusterList`：输入簇列表，每个元素`1 byte`。
- `NumOutClusters`：输出簇的个数。
- `OutClusterList`：输出簇列表，每个元素`1 byte`。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_EndDeviceBindRsp

&emsp;&emsp;这个宏定义直接调用`ZDP_SendData`，调用此函数响应终端设备绑定请求。

``` cpp
afStatus_t ZDP_EndDeviceBindRsp ( byte TranSeq, zAddrType_t *dstAddr, byte Status, byte SecurityEnable );
```

- `TranSeq`：传输序列号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态          | 数值
--------------|-----
`SUCCESS`     | `0`
`NOT_SUPPORT` | `1`
`TIMEOUT`     | `2`
`NO_MATCH`    | `3`

- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 终端设备绑定请求和响应应用举例分析

&emsp;&emsp;协调器首先需要在`ZDO`层注册`End_Device_Bind_req`信息。然后本地节点调用`ZDP_EndDeviceBindReq`发送终端设备绑定请求至协调器。协调器接收到该请求信息(该信息从属于`ZDO_CB_MSG`)，调用`ZDO_MatchEndDeviceBind`处理终端设备绑定请求。处理完毕之后，调用`ZDP_EndDeviceBindRsp`将反馈信息发送给本地节点。

#### ZDP_BindReq

&emsp;&emsp;这个宏定义实际上是调用了函数`ZDP_BindUnbindReq`，该函数将建立和发送一个绑定请求。

``` cpp
afStatus_t ZDP_BindReq (
    zAddrType_t *dstAddr,
    byte *SourceAddr,
    byte SrcEP,
    byte ClusterID,
    byte *DestinationAddr,
    byte DstEP,
    byte SecuritySuite
);
```

- `DstAddr`：目的地址。
- `SourceAddr`：本地节点`64`位`IEEE`地址。
- `SrcEP`：本地节点的端点。
- `ClusterID`：绑定簇的`ID`号。
- `DestinationAddr`：远程节点的`64`位`IEEE`地址。
- `DstEP`：远程节点的端点。
- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### ZDP_BindRsp

&emsp;&emsp;这个宏定义直接调用`ZDP_SendData`，调用此函数响应绑定请求。

``` cpp
afStatus_t ZDP_EndDeviceBindRsp ( byte TranSeq, zAddrType_t *dstAddr, byte Status, byte SecurityEnable );
```

- `TranSeq`：传输序列号。
- `DstAddr`：目的地址。
- `Status`：有如下状态：

状态          | 数值
--------------|-----
`SUCCESS`     | `0`
`NOT_SUPPORT` | `1`
`TABLE_FULL`  | `2`

- `SecuritySuite`：安全要求。

返回值`ZStatus_t`为状态。

#### 绑定请求和响应应用举例分析

&emsp;&emsp;本例中协调器欲与远程节点建立绑定连接，协调器在`ZDO`层注册`Bind_rsp`消息事件，而远程节点在`ZDO`层注册`Bind_Req`消息事件。协调器调用函数`ZDP_BindReq`发起绑定请求，远程节点接收到绑定请求后，根据`ZDO_CB_MSG`消息事件调用函数`ZDO_ProcessBindUnbindReq`处理绑定请求，建立绑定表，并且通过调用函数`ZDP_SendData`发送响应信息至协调器。协调器接收到绑定反馈消息后，根据`ZDO_CB_MSG`消息事件调用函数`ZDMatchSendState`处理绑定绑定反馈信息。

## 应用框架(AF)

&emsp;&emsp;应用框架层是`APS`层应用程序的无线数据接口。应用程序使用该层提供的功能(通过`APS`层和`NWK`层)来无线收发数据。

### 概述

&emsp;&emsp;`AF`提供以下`2`个应用：端点管理和收发数据。

### 端点管理

&emsp;&emsp;在`ZigBee`网络中，每个设备都是一个节点，每个节点具有唯一的一个`IEEE`地址(`64`位)和一个网络地址(`16`位)。网络中的其他节点发送数据时必须指定目标节点的短地址，数据才能被接收。每个节点有`241`个端点，其中端点`0`由`ZDO`层使用，它是不可缺少的。端点`1`至`240`由应用程序分配使用，在`ZigBee`网络中应用程序必须登记注册一个或多个端点，这样才能发送和接收数据。

#### afRegister

&emsp;&emsp;这个函数用来为节点登记注册一个新的端点。应用程序会为每个端点调用这个函数。

``` cpp
afStatus_t afRegister ( endPointDesc_t *epDesc );
```

参数`epDesc`为端点描述符。返回值`afStatus_t`为状态。应用举例(`ZDApp.c`)：

``` cpp
afRegister ( ( endPointDesc_t * ) &ZDApp_epDesc );
```

其中，`ZDApp_epDesc`为端点`0`的端点描述符：

``` cpp
endPointDesc_t ZDApp_epDesc = {
    ZDO_EP,
    &ZDAppTaskID,
    /* No Simple description for ZDO */
    ( SimpleDescriptionFormat_t * ) NULL,
    ( afNetworkLatencyReq_t ) 0 /* No Network Latency req */
};
```

`Sapi.c`例子如下：

``` cpp
afRegister ( &sapi_epDesc );
```

其中，`sapi_epDesc`为应用端点`2`的端点描述符：

``` cpp
sapi_epDesc.endPoint = zb_SimpleDesc.EndPoint;
sapi_epDesc.task_id = &sapi_TaskID;
sapi_epDesc.simpleDesc = ( SimpleDescriptionFormat_t * ) &zb_SimpleDesc;
sapi_epDesc.latencyReq = noLatencyReqs;
const SimpleDescriptionFormat_t zb_SimpleDesc = {
    MY_ENDPOINT_ID, /* Endpoint */
    MY_PROFILE_ID, /* Profile ID */
    DEV_ID_CONTROLLER, /* Device ID */
    DEVICE_VERSION_CONTROLLER, /* Device Version */
    0, /* Reserved */
    NUM_IN_CMD_CONTROLLER, /* Number of Input Commands */
    ( cId_t * ) zb_InCmdList, /* Input Command List */
    NUM_OUT_CMD_CONTROLLER, /* Number of Output Commands */
    ( cId_t * ) NULL /* Output Command List */
};
```

#### afFindEndPointDesc

&emsp;&emsp;这个函数用来从一个端点查找相应的端点描述符。

``` cpp
endPointDesc_t *afFindEndPointDesc ( byte endPoint );
```

参数`endPoint`为端点号。返回值`endPointDesc_t *`为端点描述符。

#### afFindSimpleDesc

&emsp;&emsp;这个函数用来从一个端点查找相应的简单描述符。

``` cpp
byte afFindSimpleDesc ( SimpleDescriptionFormat_t **ppDesc, byte EP );
```

- `ppDesc`：简单描述符。
- `EP`：端点号。

返回值为简单描述符内存地址。

#### afGetMatch

&emsp;&emsp;默认情况下，节点将响应`ZDO`匹配描述符请求，可以使用这个函数获得`ZDO`匹配描述符响应信息的设置。

``` cpp
uint8 afGetMatch ( uint8 ep );
```

参数`ep`为得到`ZDO`匹配描述符响应信息的端点号。返回值为`true`表示允许响应，为`false`表示不允许或未找到。

#### afSetMatch

&emsp;&emsp;默认情况下，节点将响应`ZDO`匹配描述符请求，可以使用这个函数改变此行为。

``` cpp
uint8 afSetMatch ( uint8 ep, uint8 action );
```

- `ep`：改变`ZDO`匹配描述符响应信息行为的端点号。
- `action`：`true`表示允许响应(默认值)，`false`表示该端点不允许响应。

返回值为`true`表示成功，为`false`表示未找到。

### 发送数据

#### AF_DataRequest

&emsp;&emsp;调用此函数发送数据。

``` cpp
afStatus_t AF_DataRequest (
    afAddrType_t *dstAddr,
    endPointDesc_t *srcEP,
    uint16 cID,
    uint16 len,
    uint8 *buf,
    uint8 *transID,
    uint8 options,
    uint8 radius
);
```

- `dstAddr`：目的地址。其中地址模式可以是：

模式               | 说明
-------------------|--------
`afAddrNotPresent` | 绑定模式地址
`afAddrGroup`      | 组播地址
`afAddrBroadcast`  | 广播地址
`afAddr16Bit`      | 直接传输(单播)地址

- `srcEP`：源端点。
- `cID`：簇`ID`号(`Cluster ID`)。
- `len`：`buf`域的长度，发送数据的字节数。
- `buf`：准备发送的数据。
- `transID`：传输序列号。
- `options`：有如下选项：

选项   | 说明
-------|-----
`0x10` | `AF_ACK_REQUEST`
`0x20` | `AF_DISCV_ROUTE`
`0x40` | `AF_EN_SECURITY`
`0x80` | `AF_SKIP_ROUTING`

- `radius`：最大的跳数。

返回值`afStatus_t`为状态。

### 接收数据

&emsp;&emsp;数据封包发送至登记注册过的端点。应用程序将通过`AF_INCOMING_MSG_CMD`的`OSAL`消息事件处理接收到的数据封包。接收到的数据封包结构如下：

``` cpp
typedef struct {
    osal_event_hdr_t hdr;     /* OSAL Message header */
    uint16 groupId;           /* Message's group ID - 0 if not set */
    uint16 clusterId;         /* Message's cluster ID */
    afAddrType_t srcAddr;     /* Source Address*/
    uint16 macDestAddr;       /* MAC header destination short address */
    uint8 endPoint;           /* destination endpoint */
    uint8 wasBroadcast;       /* TRUE if network destination was a broadcast address */
    uint8 LinkQuality;        /* The link quality of the received data frame */
    uint8 correlation;        /* The raw correlation value of the received data frame */
    int8 rssi;                /* The received RF power in units dBm */
    uint8 SecurityUse;        /* deprecated */
    uint32 timestamp;         /* receipt timestamp from MAC */
    afMSGCommandFormat_t cmd; /* Application Data */
} afIncomingMSGPacket_t;

typedef struct {
    uint8 event;
    uint8 status;
} osal_event_hdr_t;

typedef struct {
    union {
        uint16 shortAddr;
        ZLongAddr_t extAddr;
    } addr;

    afAddrMode_t addrMode;
    byte endPoint;
    uint16 panId; /* used for the INTER_PAN feature */
} afAddrType_t;

typedef enum {
    afAddrNotPresent = AddrNotPresent,
    afAddr16Bit = Addr16Bit,
    afAddr64Bit = Addr64Bit,
    afAddrGroup = AddrGroup,
    afAddrBroadcast = AddrBroadcast
} afAddrMode_t;

typedef struct {
    byte TransSeqNumber;
    uint16 DataLength; /* Number of bytes in TransData */
    byte *Data;
} afMSGCommandFormat_t;
```

## 应用支持子层(APS)

&emsp;&emsp;`APS`层提供以下管理功能：绑定表管理、组表管理、快速地址查询。

### 绑定表管理

&emsp;&emsp;绑定表定义在静态`RAM`区中。表的大小由`f8wConfig.cfg`中的`NWK_MAX_BINDING_ENTRIES`(表的条目数)和`MAX_BINDING_CLUSTER_IDS`(每个条目中簇的个数)决定，该表可以从`nwk_globals.c`中获得。在`APS`层中使用编译指令`REFLFCTOR`启用绑定功能。绑定表结构如下：

``` cpp
BindingEntry_t BindingTable[NWK_MAX_BINDING_ENTRIES]; /* Binding Table */
-DNWK_MAX_BINDING_ENTRIES = 4 /* Maximum number of entries in the Binding table. */

typedef struct {
    /* No src address since the src is always the local device */
    uint8 srcEP;
    /* Destination address type; 0 - Normal address index, 1 - Group address */
    uint8 dstGroupMode;
    /* This field is used in both modes (group and non-group) to save NV and RAM space */
    uint16 dstIdx;
    /* dstGroupMode = 0 - Address Manager index
       dstGroupMode = 1 - Group Address */
    uint8 dstEP;
    uint8 numClusterIds;
    uint16 clusterIdList[MAX_BINDING_CLUSTER_IDS];
} BindingEntry_t;

-DMAX_BINDING_CLUSTER_IDS = 4
```

### 组表管理

&emsp;&emsp;组表是一个分配在`RAM[osal_mem_alloc]`中的一个链表，因此随着表中组个数增加，所分配的堆栈也随之增长。表的最大尺寸由定义在`f8wConfig.cfg`中的`APS_MAX_GROUPS`确定。组表可以从`nwk_globals.c`中获得。

``` cpp
typedef struct apsGroupItem {
    struct apsGroupItem *next;
    uint8 endpoint;
    aps_Group_t group;
} apsGroupItem_t;

typedef struct {
    uint16 ID; /* Unique to this table */
    uint8 name[APS_GROUP_NAME_LEN]; /* Human readable name of group */
} aps_Group_t;
```

#### aps_AddGroup

&emsp;&emsp;调用此函数可以增加一个组至组表中。

``` cpp
ZStatus_t aps_AddGroup ( uint8 endpoint, aps_Group_t *group );
```

- `endpoint`：新增组的端点。
- `group`：增加至组表的组号和组名。

返回值如下：

返回值                                             | 说明
---------------------------------------------------|-----
`ZSuccess`                                         | 成功
`ZApsDuplicateEntry`、`ZApsTableFull`或`ZMemError` | 失败

#### aps_RemoveGroup

&emsp;&emsp;调用此函数可以从组表中删除一个组。

``` cpp
ZStatus_t aps_RemoveGroup ( uint8 endpoint, uint16 groupID );
```

- `endpoint`：删除组的端点。
- `groupID`：被删除组的组号。

若成功，则函数返回`true`；失败则返回`false`。

#### aps_FindGroup

&emsp;&emsp;调用此函数可以根据端点和组号从组表中查找到一个组。

``` cpp
aps_Group_t *aps_FindGroup ( uint8 endpoint, uint16 groupID );
```

- `endpoint`：查找组的端点。
- `groupID`：查找组的组号。

若找到，则函数指向组的指针；未找到则返回`NULL`。

## 网络层(NWK)

&emsp;&emsp;网络层为上层提供如下功能：网络管理、地址管理、网络参数与功能函数。

### 网络管理

#### NLME_NetworkFormationRequest

&emsp;&emsp;此函数请求组建一个新网络并允许自身成为该网络的`ZigBee`协调器。这个行为的结果返回到`ZDO_NetworkFormationConfirmCB`中。

``` cpp
ZStatus_t NLME_NetworkFormationRequest (
    uint16 PanId,
    uint8 *ExtendedPANID,
    uint32 ScanChannels,
    byte ScanDuration,
    byte BeaconOrder,
    byte SuperframeOrder,
    byte BatteryLifeExtension
);
```

- `PanId`：个域网`ID`号，有效值为`0`至`0xFFFE`。如果为`0xFFFF`，则网络层将选择`PANID`供网络使用。
- `ExtendedPANID`：扩展的个域网`ID`号。
- `ScanChannels`：扫描的频道，其值为`11`至`26`。
- `ScanDuration`：扫描时间。
- `BeaconOrder`：该值为`BEACON_ORDER_NO_BEACONS`。
- `SuperframeOrder`：该值为`BEACON_ORDER_NO_BEACONS`。
- `BatteryLifeExtension`：电池寿命延长模式。

返回值`ZStatus_t`为状态。

#### 网络层：组建网络请求举例分析

&emsp;&emsp;调用组建网络请求函数的必要条件是：`logicalType == NODETYPE_COORDINATOR`并且`startMode == MODE_HARD`。本例中`NLME_NetworkFormationRequest`各参数设置如下(在`f8wConfig.cfg`中定义)：

``` cpp
-DZDAPP_CONFIG_PAN_ID=0xFFFF
uint16 zgConfigPANID = ZDAPP_CONFIG_PAN_ID;
uint8 zgApsUseExtendedPANID[Z_EXTADDR_LEN] = {00, 00, 00, 00, 00, 00, 00, 00};
-DDEFAULT_CHANLIST=0x00000800 /* 11 - 0x0B */
#define STARTING_SCAN_DURATION 5
uint8 zgDefaultStartingScanDuration = STARTING_SCAN_DURATION;
#define BEACON_ORDER_NO_BEACONS 15
beaconOrder = BEACON_ORDER_NO_BEACONS;
superFrameOrder = BEACON_ORDER_NO_BEACONS;
```

返回函数`ZDO_NetworkFormationConfirmCB`将设置网络启动事件，交给操作系统做下一步处理：

``` cpp
void ZDO_NetworkFormationConfirmCB ( ZStatus_t Status ) {
    osal_set_event ( ZDAppTaskID, ZDO_NETWORK_START );
}
```

组建网络请求和反馈发生在协调器的`ZDO`层与`NWK`层之间：

#### NLME_NetworkDiscoveryRequest

&emsp;&emsp;此函数请求网络层寻找相邻的路由器。这个函数应该在加入并执行网络扫描之前调用。这个行为的结果返回到`ZDO_NetworkDiscoveryConfirmCB`中，返回结果包含：发现的路由个数和网络描述列表。

``` cpp
ZStatus_t NLME_NetworkDiscoveryRequest ( uint32 ScanChannels, byte ScanDuration );
```

- `ScanChannels`：扫描的频道，其值为`11`至`26`。
- `ScanDuration`：扫描时间。

返回值`ZStatus_t`为状态。

#### 网络层：发现网络请求举例分析

&emsp;&emsp;发现网络请求举例分析调用发现网络请求函数的必要条件是：`logicalType == NODETYPE_ROUTER || logicalType == NODETYPE_DEVICE`并且`startMode == MODE_JOIN || startMode == MODE_REJOIN`。本例中`NLME_NetworkDiscoveryRequest`各参数设置如下(在`f8wConfig.cfg`中定义)：

``` cpp
DDEFAULT_CHANLIST=0x00000800 /* 11 - 0x0B */
#define STARTING_SCAN_DURATION 5
uint8 zgDefaultStartingScanDuration = STARTING_SCAN_DURATION;
```

返回函数`ZDO_NetworkDiscoveryConfirmCB`中网络描述列表定义如下：

``` cpp
typedef struct {
    uint16 panId;
    byte logicalChannel;
    byte beaconOrder;
    byte superFrameOrder;
    byte routerCapacity;
    byte deviceCapacity;
    byte version;
    byte stackProfile;
    uint16 chosenRouter;
    uint8 chosenRouterLinkQuality;
    uint8 chosenRouterDepth;
    uint8 extendedPANID[Z_EXTADDR_LEN];
    byte updateId;
    void *nextDesc;
} networkDesc_t;
```

该函数将网络列表的部分元素传递至发现网络反馈消息`ZDO_NetworkDiscoveryCfm_t`，并设置发现网络消息事件`ZDO_NWK_DISC_CNF`，交由`ZDApp`任务事件处理函数做下一步处理：

``` cpp
typedef struct {
    osal_event_hdr_t hdr;
    uint8 panIdLSB;
    uint8 panIdMSB;
    uint8 logicalChannel;
    uint8 version;
    uint8 extendedPANID[Z_EXTADDR_LEN];
} ZDO_NetworkDiscoveryCfm_t;

ZStatus_t ZDO_NetworkDiscoveryConfirmCB ( uint8 ResultCount, networkDesc_t *NetworkList ) {
    networkDesc_t *pNwkDesc = NetworkList;
    msg.hdr.status = ZDO_SUCCESS;
    msg.panIdLSB = LO_UINT16 ( pNwkDesc->panId );
    msg.panIdMSB = HI_UINT16 ( pNwkDesc->panId );
    msg.logicalChannel = pNwkDesc->logicalChannel;
    msg.version = pNwkDesc->version;
    osal_cpyExtAddr ( msg.extendedPANID, pNwkDesc->extendedPANID );
    ZDApp_SendMsg ( ZDAppTaskID, ZDO_NWK_DISC_CNF, sizeof ( ZDO_NetworkDiscoveryCfm_t ), ( uint8 * ) &msg );
}
```

发现网络请求和反馈发生在路由器或终端设备的`ZDO`层与`NWK`层之间：

#### NLME_JoinRequest

&emsp;&emsp;此函数请求节点将自己加入到一个网络中。这个行为的结果返回到`ZDO_JoinConfirmCB`中。

``` cpp
ZStatus_t NLME_JoinRequest ( uint8 *ExtendedPANID, uint16 PanId, byte Channel, byte CapabilityInfo );
```

- `ExtendedPANID`：试图加入的网络的扩展`PAN ID`号。
- `PanId`：试图加入的网络的`PAN ID`号。
- `Channel`：频道号，其值为`11`至`26`。
- `CapabilityInfo`：正在加入网络的设备的能力。

返回值`ZStatus_t`为状态。

#### 网络层：加入网络请求举例分析

&emsp;&emsp;调用加入网络请求函数的必要条件是：`devStartMode == MODE_JOIN`。本例中`NLME_JoinRequest`各参数设置如下：

``` cpp
( ZDO_NetworkDiscoveryCfm_t * ) msgPtr )->extendedPANID  /* 试图加入的网络的扩展“PAN ID”号  */
( ZDO_NetworkDiscoveryCfm_t * ) msgPtr )->panIdLSB       /* 试图加入的网络的"PAN ID"号高8位 */
( ZDO_NetworkDiscoveryCfm_t * ) msgPtr )->panIdMSB       /* 试图加入的网络的"PAN ID"号低8位 */
( ZDO_NetworkDiscoveryCfm_t * ) msgPtr )->logicalChannel /* 频道号                         */
ZDO_Config_Node_Descriptor.CapabilityFlags               /* 节点描述符中的节点能力          */
```

返回函数`ZDO_JoinConfirmCB`设置加入网络消息事件`ZDO_NWK_JOIN_IND`，交由`ZDApp`任务事件处理函数做下一步处理：

``` cpp
void ZDO_JoinConfirmCB ( uint16 PanId, ZStatus_t Status ) {
    ZDApp_SendMsg ( ZDAppTaskID, ZDO_NWK_JOIN_IND, sizeof ( osal_event_hdr_t ), ( byte * ) NULL );
}
```

加入网络请求和反馈发生在路由器或终端设备的`ZDO`层与`NWK`层之间：

#### NLME_ReJoinRequest

&emsp;&emsp;此函数请求节点重新加入到一个已经加入过的网络中。这个行为的结果返回到`ZDO_JoinConfirmCB`中。

``` cpp
ZStatus_t NLME_ReJoinRequest ( uint8 *ExtendedPANID, byte Channel );
```

- `ExtendedPANID`：试图加入的网络的扩展`PAN ID`号。
- `Channel`：频道号，其值为`11`至`26`。

返回值`ZStatus_t`为状态。

#### 网络层：重新加入网络请求举例分析

&emsp;&emsp;调用重新加入网络请求函数的必要条件是：`devStartMode == MODE_ReJOIN`。本例中`NLME_JoinRequest`各参数设置如下：

``` cpp
ZDO_UseExtendedPANID: /* 试图加入的网络的扩展“PAN ID”号 */
( ZDO_NetworkDiscoveryCfm_t * ) msgPtr )->logicalChannel /* 频道号 */
```

返回函数`ZStatus_t ZDO_JoinConfirmCB`设置加入网络消息事件`ZDO_NWK_JOIN_IND`，交由ZDApp任务事件处理函数做下一步处理：

``` cpp
void ZDO_JoinConfirmCB ( uint16 PanId, ZStatus_t Status ) {
    ZDApp_SendMsg ( ZDAppTaskID, ZDO_NWK_JOIN_IND, sizeof ( osal_event_hdr_t ), ( byte * ) NULL );
}
```

重新加入网络请求和反馈发生在路由器或终端设备的`ZDO`层与`NWK`层之间：

#### NLME_OrphanJoinRequest

&emsp;&emsp;此函数请求孤立节点连接到一个父节点。这个行为的结果返回到`ZDO_JoinConfirmCB`中。

``` cpp
ZStatus_t NLME_OrphanJoinRequest ( uint32 ScanChannels, byte ScanDuration );
```

- `ScanChannels`：频道号，其值为`11`至`26`。
- `ScanDuration`：扫描时间。

返回值`ZStatus_t`为状态。

#### 网络层：孤立节点连接父节点请求举例分析

&emsp;&emsp;调用孤立节点连接父节点请求函数的必要条件是：`startMode == MODE_RESUME`并且`logicalType == NODETYPE_DEVICE`。本例中`NLME_OrphanJoinRequest`各参数设置如下：

``` cpp
zgDefaultChannelList = DEFAULT_CHANLIST /* 频道号 = 11 */
zgDefaultStartingScanDuration = STARTING_SCAN_DURATION /* 扫描时间 = 5 */
```

返回函数`ZStatus_t ZDO_JoinConfirmCB`设置加入网络消息事件`ZDO_NWK_JOIN_IND`，交由`ZDApp`任务事件处理函数做下一步处理：

``` cpp
void ZDO_JoinConfirmCB ( uint16 PanId, ZStatus_t Status ) {
    ZDApp_SendMsg ( ZDAppTaskID, ZDO_NWK_JOIN_IND, sizeof ( osal_event_hdr_t ), ( byte * ) NULL );
}
```

孤立节点连接父节点请求和反馈发生在路由器或终端设备的`ZDO`层与`NWK`层之间：

#### NLME_StartRouterRequest

&emsp;&emsp;此函数请求启动路由器。这个行为的结果返回到`ZDO_StartRouterConfirmCB`中。

``` cpp
ZStatus_t NLME_StartRouterRequest ( byte BeaconOrder, byte SuperframeOrder, byte BatteryLifeExtension );
```

- `BeaconOrder`：`BEACON_ORDER_NO_BEACONS`。
- `SuperframeOrder`：`BEACON_ORDER_NO_BEACONS`。
- `BatteryLifeExtension`：`TRUE`为电池供电，`FALSE`为电源供电。

返回值`ZStatus_t`为状态。

#### 网络层：路由器启动请求举例分析

&emsp;&emsp;调用路由器启动请求函数的必要条件是：`logicalType == NODETYPE_COORDINATOR`并且`startMode == MODE_RESUME`，或者`LogicalType == NODETYPE_ROUTER`并且`devState == DEV_NWK_JOINING || devState == DEV_NWK_ORPHAN || devState == DEV_NWK_REJOIN`。本例中`NLME_StartRouterRequest`各参数设置如下：

``` cpp
BeaconOrder=0
SuperframeOrder=0
BatteryLifeExtension=False
```

返回函数`ZStatus_t ZDO_StartRouterConfirmCB`设置路由启动事件`ZDO_ROUTER_START`，交由`ZDApp`任务事件处理函数做下一步处理：

``` cpp
void ZDO_StartRouterConfirmCB ( ZStatus_t Status ) {
    osal_set_event ( ZDAppTaskID, ZDO_ROUTER_START );
}
```

路由启动请求和反馈发生在路由器或终端设备的`ZDO`层与`NWK`层之间：

### 地址管理

&emsp;&emsp;查询并存储在本地设备的远程地址。

#### NLME_GetExtAddr

&emsp;&emsp;调用此函数得到节点自身的`64`位`IEEE`地址。

``` cpp
byte *NLME_GetExtAddr ( void );
```

返回值为指向`64`位`IEEE`地址的指针。

#### NLME_GetShortAddr

&emsp;&emsp;调用此函数得到节点自身的`16`位网络地址。

``` cpp
byte *NLME_GetShortAddr ( void );
```

返回值为`16`位网络地址。

#### NLME_GetCoordShortAddr

&emsp;&emsp;调用此函数得到父节点的`16`位网络地址。

``` cpp
byte *NLME_GetCoordShortAddr ( void );
```

返回值为父节点`16`位网络地址。

#### NLME_GetCoordExtAddr

&emsp;&emsp;调用此函数得到父节点的`64`位`IEEE`地址。

``` cpp
byte *NLME_GetCoordExtAddr ( void );
```

返回值为指向父节点`64`位`IEEE`地址的指针。

#### NLME_IsAddressBroadcast

&emsp;&emsp;此函数根据设备能力来评估提供的地址是否是一个有效的广播地址。

``` cpp
addr_filter_t NLME_IsAddressBroadcast ( uint16 shortAddress );
```

参数`shortAddress`为被测试的`16`位网络地址。返回值`addr_filter_t`为`ADDR_NOT_BCAST`、`ADDR_BCAST_FOR_ME`、`ADDR_BCAST_NOT_ME`。

#### NLME_SetBroadCastFilter

&emsp;&emsp;此函数根据设备能力设置掩码，用于处理有效的广播地址。

``` cpp
void NLME_SetBroadcastFilter ( byte capabilities );
```

参数`capabilities`为该设备的能力用于确定该设备处理的广播信息类型。

---

# ZDO解析函数

&emsp;&emsp;`ZDO`解析函数用来解析接收到的消息。

``` cpp
ZDO_NwkIEEEAddrResp_t *ZDO_ParseAddrRsp ( zdoIncomingMsg_t *inMsg );
```

用于解析`NWK_addr_rsp`和`IEEE_addr_rsp`消息。参数`inMsg`指向接收到的消息的指针，返回值`ZDO_NwkIEEEAddrResp_t`指向解析后的结构的指针，结构体由`osal_mem_alloc`分配空间，所以需要调用`osal_mem_free`来释放空间。

``` cpp
void ZDO_ParseNodeDescRsp ( zdoIncomingMsg_t *inMsg, ZDO_NodeDescRsp_t *pNDRsp );
```

用于解析`Node_Desc_rsp`消息。参数`pNDRsp`是解析消息存放的地方。

``` cpp
void ZDO_ParsePowerDescRsp ( zdoIncomingMsg_t *inMsg, ZDO_PowerRsp_t *pNPRsp );
```

用于解析`Power_Desc_rsp`消息。

``` cpp
void ZDO_ParseSimpleDescRsp ( zdoIncomingMsg_t *inMsg, ZDO_SimpleDescRsp_t *pSimpleDescRsp );
```

用于解析`SimpleDescRsp`消息，参数`pSimpleDescRsp`用于存放解析后的消息。

``` cpp
ZDO_ActiveEndpointRsp_t *ZDO_ParseEPListRsp ( zdoIncomingMsg_t *inMsg );
```

用于解析`Active_EP_rsp`或者`Match_Desc_rsp`的消息。

``` cpp
#define ZDO_ParseBindRsp(a) ((uint8)(*(a->asdu)))
```

用于解析`Bind_rsp`、`Unbind_rsp`或`End_Device_Bind_rsp`消息。参数`a`指向要解析的消息的指针。

``` cpp
ZDO_MgmNwkDiscRsp_t *ZDO_ParseMgmNwkDiscRsp ( zdoIncomingMsg_t *inMsg );
```

解析`Mgmt_NWK_Disc_rsp`消息。