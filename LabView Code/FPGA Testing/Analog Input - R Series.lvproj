<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="16008000">
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">false</Property>
	<Property Name="NI.LV.ExampleFinder" Type="Str">&lt;?xml version="1.0" encoding="UTF-8"?&gt;
&lt;ExampleProgram&gt;
&lt;Title&gt;
	&lt;Text Locale="US"&gt;Analog Input - R Series.lvproj&lt;/Text&gt;
&lt;/Title&gt;
&lt;Keywords&gt;
	&lt;Item&gt;FPGA&lt;/Item&gt;
	&lt;Item&gt;started&lt;/Item&gt;
	&lt;Item&gt;getting&lt;/Item&gt;
	&lt;Item&gt;analog&lt;/Item&gt;
	&lt;Item&gt;input&lt;/Item&gt;
	&lt;Item&gt;7830&lt;/Item&gt;
	&lt;Item&gt;7831&lt;/Item&gt;
	&lt;Item&gt;7833&lt;/Item&gt;
	&lt;Item&gt;7841&lt;/Item&gt;
	&lt;Item&gt;7842&lt;/Item&gt;
	&lt;Item&gt;7851&lt;/Item&gt;
	&lt;Item&gt;7852&lt;/Item&gt;
	&lt;Item&gt;7853&lt;/Item&gt;
	&lt;Item&gt;7854&lt;/Item&gt;
	&lt;Item&gt;7855&lt;/Item&gt;
	&lt;Item&gt;7856&lt;/Item&gt;
	&lt;Item&gt;Input&lt;/Item&gt;
	&lt;Item&gt;7846&lt;/Item&gt;
	&lt;Item&gt;7845&lt;/Item&gt;
	&lt;Item&gt;7847&lt;/Item&gt;
	&lt;Item&gt;7857&lt;/Item&gt;
	&lt;Item&gt;7858&lt;/Item&gt;
&lt;/Keywords&gt;
&lt;Navigation&gt;
	&lt;Item&gt;7253&lt;/Item&gt;
	&lt;Item&gt;7695&lt;/Item&gt;
&lt;/Navigation&gt;
&lt;FileType&gt;LV Project&lt;/FileType&gt;
&lt;Metadata&gt;
&lt;Item Name="RTSupport"&gt;LV Project RT&lt;/Item&gt;
&lt;/Metadata&gt;
&lt;ProgrammingLanguages&gt;
&lt;Item&gt;LabVIEW&lt;/Item&gt;
&lt;/ProgrammingLanguages&gt;
&lt;RequiredSoftware&gt;
&lt;NiSoftware MinVersion="11.0"&gt;LabVIEW&lt;/NiSoftware&gt; 
&lt;/RequiredSoftware&gt;
&lt;RequiredFPGAHardware&gt;
&lt;Device&gt;
&lt;Model&gt;7055&lt;/Model&gt;
&lt;Model&gt;7056&lt;/Model&gt;
&lt;Model&gt;702C&lt;/Model&gt;
&lt;Model&gt;702D&lt;/Model&gt;
&lt;Model&gt;7074&lt;/Model&gt;
&lt;Model&gt;7083&lt;/Model&gt;
&lt;Model&gt;7390&lt;/Model&gt;
&lt;Model&gt;7391&lt;/Model&gt;
&lt;Model&gt;7384&lt;/Model&gt;
&lt;Model&gt;7385&lt;/Model&gt;
&lt;Model&gt;7392&lt;/Model&gt;
&lt;Model&gt;73E1&lt;/Model&gt;
&lt;Model&gt;76DE&lt;/Model&gt;
&lt;Model&gt;76E0&lt;/Model&gt;
&lt;Model&gt;76DF&lt;/Model&gt;
&lt;Model&gt;76E1&lt;/Model&gt;
&lt;Model&gt;776B&lt;/Model&gt;
&lt;Model&gt;776C&lt;/Model&gt;
&lt;Model&gt;776D&lt;/Model&gt;
&lt;Model&gt;776E&lt;/Model&gt;
&lt;Model&gt;783B&lt;/Model&gt;
&lt;Model&gt;783A&lt;/Model&gt;
&lt;Model&gt;7839&lt;/Model&gt;
&lt;Model&gt;7838&lt;/Model&gt;
&lt;Model&gt;7837&lt;/Model&gt;
&lt;/Device&gt;
&lt;/RequiredFPGAHardware&gt;
&lt;/ExampleProgram&gt;</Property>
	<Property Name="NI.Project.Description" Type="Str">This project shows how to read analog input channels from an R Series board.

This example needs to be compiled for a specific FPGA target before use. For information on moving this example to another FPGA target, refer to ni.com/info and enter info code fpgaex.</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="CCSymbols" Type="Str">OS,Win;CPU,x86;</Property>
		<Property Name="IOScan.Faults" Type="Str"></Property>
		<Property Name="IOScan.NetVarPeriod" Type="UInt">100</Property>
		<Property Name="IOScan.NetWatchdogEnabled" Type="Bool">false</Property>
		<Property Name="IOScan.Period" Type="UInt">10000</Property>
		<Property Name="IOScan.PowerupMode" Type="UInt">0</Property>
		<Property Name="IOScan.Priority" Type="UInt">9</Property>
		<Property Name="IOScan.ReportModeConflict" Type="Bool">true</Property>
		<Property Name="IOScan.StartEngineOnDeploy" Type="Bool">false</Property>
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">3363</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="FPGA Target 2" Type="FPGA Target">
			<Property Name="AutoRun" Type="Bool">false</Property>
			<Property Name="configString.guid" Type="Str">{008CD651-275E-447A-BA70-735F3B420759}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{038E7F04-5E3B-4603-AC36-C8CE4A6DFC4D}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AI2;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{1041F308-37C6-4C5E-A3EE-94C02F8E0161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{12B90C11-C121-4DFE-8C62-24E51442255D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{14A058DB-8172-4F9F-AFD2-28B18CE983CD}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{221E1121-D533-4CBF-A737-F1BB6712C581}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AI3;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{2B99CF6B-2071-45BA-89FA-CF6599354F78}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AO2;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{2CF5EC02-07DF-4E72-8155-734D002B89E6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{336542BA-C4B3-466E-96B5-412732D2847A}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AI0;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{35828FAE-164E-4516-BF4C-AFF0258177FF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT1;0;ReadMethodType=u8;WriteMethodType=u8{3A9E951A-AE12-4C8A-94F2-C77291F35B75}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AI4;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{409CC936-3F10-4BDC-B254-503FF7C35D42}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AO5;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{45D4AF25-55BB-481D-ADD0-492417651E17}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AO4;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{477DE55D-EF4D-4E36-8937-DB97FB3055D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{4796C5C0-78DF-4F3E-974F-512CC04088B4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{4DFF36D0-0823-417D-BAFA-8182BCA5E678}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AO1;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{5EE331A7-B778-4C67-BFEF-3396249F1546}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{76FB1159-ED97-4482-BB50-59D5F795E124}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{7DB8D115-4F08-44B5-B4E8-AC7D7CFD98E5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{8080AA8A-5AF8-498A-81B5-49BB3B69BC83}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AI6;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{A090ECDA-5632-4CED-BB71-F67DB9C01993}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AO6;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{A5F733FD-1CDE-47D7-BA19-13999A7BCF56}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{AB14BCF9-93C1-449F-91CF-4E51B0A321F9}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AO3;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{B0769666-2416-467B-A85E-A94CC1326A76}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AO7;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{BC5F0CD1-3999-4356-BA19-D2DE22F8D951}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{C01E16FD-B82D-4584-987B-73C1ECB89310}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AO0;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{C0650B72-BF16-4B44-B9FC-9ABDB30A893A}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AI7;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{CB0C98F0-C19C-47ED-AE3A-DFB7FAF5B043}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D310D2A0-2282-4394-A358-415249206D85}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{D9918956-00C6-4FC9-A496-90EED8EF1829}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{DABF13AA-366A-4299-9E2F-4466AAAD7E28}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{E3962CEA-DFFC-41B1-A2D2-B21EEFD72DEE}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AI5;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{EFEC3C41-7BCA-48A2-9C6A-F5D497C4BAA5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{F1F018F9-263A-4ABD-A2B4-E9B04EB77815}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AI1;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{FDC317AD-4D02-4266-A1D4-3F9806FCB423}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT0;0;ReadMethodType=u8;WriteMethodType=u8PXIe-7847R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSPXIE_7847RFPGA_TARGET_FAMILYKINTEX7TARGET_TYPEFPGA/[rSeriesConfig.Begin]rseries.aio./Connector0/AI0=0,rseries.aio./Connector0/AI1=0,rseries.aio./Connector0/AI2=0,rseries.aio./Connector0/AI3=0,rseries.aio./Connector0/AI4=0,rseries.aio./Connector0/AI5=0,rseries.aio./Connector0/AI6=0,rseries.aio./Connector0/AI7=0,rseries.analogCalibratedType=1,rseries.dio./Connector0=0,rseries.dio./Connector1=0,rseries.terminalConfig=0[rSeriesConfig.End]</Property>
			<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;Connector0/AI0kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AI0;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI1kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AI1;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI2kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AI2;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI3kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AI3;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI4kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AI4;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI5kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AI5;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI6kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AI6;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI7kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AI7;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AO0kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AO0;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO1kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AO1;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO2kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AO2;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO3kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AO3;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO4kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AO4;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO5kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AO5;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO6kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AO6;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO7kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AO7;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO0;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO10;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO11;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO12;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO13;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO14;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO15;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO1;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO2;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO3;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO4;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO5;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO6;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO7;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO8;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO9;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIOPORT0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT0;0;ReadMethodType=u8;WriteMethodType=u8Connector0/DIOPORT1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT1;0;ReadMethodType=u8;WriteMethodType=u8PXIe-7847R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSPXIE_7847RFPGA_TARGET_FAMILYKINTEX7TARGET_TYPEFPGA/[rSeriesConfig.Begin]rseries.aio./Connector0/AI0=0,rseries.aio./Connector0/AI1=0,rseries.aio./Connector0/AI2=0,rseries.aio./Connector0/AI3=0,rseries.aio./Connector0/AI4=0,rseries.aio./Connector0/AI5=0,rseries.aio./Connector0/AI6=0,rseries.aio./Connector0/AI7=0,rseries.analogCalibratedType=1,rseries.dio./Connector0=0,rseries.dio./Connector1=0,rseries.terminalConfig=0[rSeriesConfig.End]</Property>
			<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">PXIe-7847R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSPXIE_7847RFPGA_TARGET_FAMILYKINTEX7TARGET_TYPEFPGA</Property>
			<Property Name="NI.LV.FPGA.Version" Type="Int">6</Property>
			<Property Name="Resource Name" Type="Str">PXI1Slot4</Property>
			<Property Name="rseries.aio./Connector0/AI0" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI1" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI2" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI3" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI4" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI5" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI6" Type="Str">0</Property>
			<Property Name="rseries.aio./Connector0/AI7" Type="Str">0</Property>
			<Property Name="rseries.analogCalibratedType" Type="Str">1</Property>
			<Property Name="rseries.dio./Connector0" Type="Str">0</Property>
			<Property Name="rseries.dio./Connector1" Type="Str">0</Property>
			<Property Name="rseries.terminalConfig" Type="Str">0</Property>
			<Property Name="Target Class" Type="Str">PXIe-7847R</Property>
			<Property Name="Top-Level Timing Source" Type="Str">40 MHz Onboard Clock</Property>
			<Property Name="Top-Level Timing Source Is Default" Type="Bool">true</Property>
			<Item Name="Connector0" Type="Folder">
				<Item Name="Connector0/AI0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{336542BA-C4B3-466E-96B5-412732D2847A}</Property>
				</Item>
				<Item Name="Connector0/AI1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>1</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{F1F018F9-263A-4ABD-A2B4-E9B04EB77815}</Property>
				</Item>
				<Item Name="Connector0/AI2" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>2</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI2</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{038E7F04-5E3B-4603-AC36-C8CE4A6DFC4D}</Property>
				</Item>
				<Item Name="Connector0/AI3" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>3</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI3</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{221E1121-D533-4CBF-A737-F1BB6712C581}</Property>
				</Item>
				<Item Name="Connector0/AI4" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>4</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI4</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{3A9E951A-AE12-4C8A-94F2-C77291F35B75}</Property>
				</Item>
				<Item Name="Connector0/AI5" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>5</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI5</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{E3962CEA-DFFC-41B1-A2D2-B21EEFD72DEE}</Property>
				</Item>
				<Item Name="Connector0/AI6" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>6</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI6</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{8080AA8A-5AF8-498A-81B5-49BB3B69BC83}</Property>
				</Item>
				<Item Name="Connector0/AI7" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntDelayInClkTicks">
   <Value>38</Value>
   </Attribute>
   <Attribute name="kIntInitGain">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>7</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI7</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{C0650B72-BF16-4B44-B9FC-9ABDB30A893A}</Property>
				</Item>
				<Item Name="Connector0/AO0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{C01E16FD-B82D-4584-987B-73C1ECB89310}</Property>
				</Item>
				<Item Name="Connector0/AO1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>1</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{4DFF36D0-0823-417D-BAFA-8182BCA5E678}</Property>
				</Item>
				<Item Name="Connector0/AO2" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>2</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO2</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{2B99CF6B-2071-45BA-89FA-CF6599354F78}</Property>
				</Item>
				<Item Name="Connector0/AO3" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>3</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO3</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{AB14BCF9-93C1-449F-91CF-4E51B0A321F9}</Property>
				</Item>
				<Item Name="Connector0/AO4" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>4</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO4</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{45D4AF25-55BB-481D-ADD0-492417651E17}</Property>
				</Item>
				<Item Name="Connector0/AO5" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>5</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO5</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{409CC936-3F10-4BDC-B254-503FF7C35D42}</Property>
				</Item>
				<Item Name="Connector0/AO6" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>6</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO6</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{A090ECDA-5632-4CED-BB71-F67DB9C01993}</Property>
				</Item>
				<Item Name="Connector0/AO7" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="kIntCalDebugEn">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntCalibrateData">
   <Value>0</Value>
   </Attribute>
   <Attribute name="kIntResChannel">
   <Value>7</Value>
   </Attribute>
   <Attribute name="kIntSimEnable">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO7</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{B0769666-2416-467B-A85E-A94CC1326A76}</Property>
				</Item>
				<Item Name="Connector0/DIOPORT0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIOPORT0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{FDC317AD-4D02-4266-A1D4-3F9806FCB423}</Property>
				</Item>
				<Item Name="Connector0/DIOPORT1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIOPORT1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{35828FAE-164E-4516-BF4C-AFF0258177FF}</Property>
				</Item>
				<Item Name="Connector0/DIO0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{DABF13AA-366A-4299-9E2F-4466AAAD7E28}</Property>
				</Item>
				<Item Name="Connector0/DIO1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{2CF5EC02-07DF-4E72-8155-734D002B89E6}</Property>
				</Item>
				<Item Name="Connector0/DIO2" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO2</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{D9918956-00C6-4FC9-A496-90EED8EF1829}</Property>
				</Item>
				<Item Name="Connector0/DIO3" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO3</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{5EE331A7-B778-4C67-BFEF-3396249F1546}</Property>
				</Item>
				<Item Name="Connector0/DIO4" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO4</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{4796C5C0-78DF-4F3E-974F-512CC04088B4}</Property>
				</Item>
				<Item Name="Connector0/DIO5" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO5</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{12B90C11-C121-4DFE-8C62-24E51442255D}</Property>
				</Item>
				<Item Name="Connector0/DIO6" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO6</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{76FB1159-ED97-4482-BB50-59D5F795E124}</Property>
				</Item>
				<Item Name="Connector0/DIO7" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO7</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{CB0C98F0-C19C-47ED-AE3A-DFB7FAF5B043}</Property>
				</Item>
				<Item Name="Connector0/DIO8" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO8</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{BC5F0CD1-3999-4356-BA19-D2DE22F8D951}</Property>
				</Item>
				<Item Name="Connector0/DIO9" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO9</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{EFEC3C41-7BCA-48A2-9C6A-F5D497C4BAA5}</Property>
				</Item>
				<Item Name="Connector0/DIO10" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO10</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{D310D2A0-2282-4394-A358-415249206D85}</Property>
				</Item>
				<Item Name="Connector0/DIO11" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO11</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{A5F733FD-1CDE-47D7-BA19-13999A7BCF56}</Property>
				</Item>
				<Item Name="Connector0/DIO12" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO12</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{477DE55D-EF4D-4E36-8937-DB97FB3055D4}</Property>
				</Item>
				<Item Name="Connector0/DIO13" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO13</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{7DB8D115-4F08-44B5-B4E8-AC7D7CFD98E5}</Property>
				</Item>
				<Item Name="Connector0/DIO14" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO14</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{1041F308-37C6-4C5E-A3EE-94C02F8E0161}</Property>
				</Item>
				<Item Name="Connector0/DIO15" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/DIO15</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{008CD651-275E-447A-BA70-735F3B420759}</Property>
				</Item>
			</Item>
			<Item Name="Analog Input.vi" Type="VI" URL="../Analog Input.vi">
				<Property Name="BuildSpec" Type="Str">{5C989541-60FF-447A-B015-8A5C02773AC3}</Property>
				<Property Name="configString.guid" Type="Str">{39908DFA-A668-4395-8ACD-AD649AE9C542}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EPCIe-7841R/Clk40/falsetrueFPGA_EXECUTION_MODEDEV_COMPUTER_SIM_IOFPGA_TARGET_CLASSPCIE_7841RFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA/[rSeriesConfig.Begin][rSeriesConfig.End]</Property>
				<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EPCIe-7841R/Clk40/falsetrueFPGA_EXECUTION_MODEDEV_COMPUTER_SIM_IOFPGA_TARGET_CLASSPCIE_7841RFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA/[rSeriesConfig.Begin][rSeriesConfig.End]</Property>
				<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\UsphereLab\Documents\GitHub\LabView Code\FPGA_Bitfiles\AnalogInput-FPGA-test.lvbitx</Property>
			</Item>
			<Item Name="RIO-DRAM" Type="FPGA Component Level IP">
				<Property Name="NI.LV.CLIP.DeclarationCategory" Type="Str"></Property>
				<Property Name="NI.LV.CLIP.SocketedCLIP" Type="Bool">true</Property>
				<Property Name="NI.LV.CLIP.SocketSelection" Type="Str">RIO-DRAM</Property>
				<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
				<Property Name="NI.SortType" Type="Int">3</Property>
			</Item>
			<Item Name="40 MHz Onboard Clock" Type="FPGA Base Clock">
				<Property Name="FPGA.PersistentID" Type="Str">{14A058DB-8172-4F9F-AFD2-28B18CE983CD}</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig" Type="Str">ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.Accuracy" Type="Dbl">100</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.ClockSignalName" Type="Str">Clk40</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MaxDutyCycle" Type="Dbl">50</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MaxFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MinDutyCycle" Type="Dbl">50</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MinFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.NominalFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.PeakPeriodJitter" Type="Dbl">250</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.ResourceName" Type="Str">40 MHz Onboard Clock</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.SupportAndRequireRuntimeEnableDisable" Type="Bool">false</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.TopSignalConnect" Type="Str">Clk40</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.VariableFrequency" Type="Bool">false</Property>
				<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
				<Property Name="NI.LV.FPGA.Version" Type="Int">5</Property>
			</Item>
			<Item Name="IP Builder" Type="IP Builder Target">
				<Item Name="Dependencies" Type="Dependencies"/>
				<Item Name="Build Specifications" Type="Build"/>
			</Item>
			<Item Name="Dependencies" Type="Dependencies">
				<Item Name="vi.lib" Type="Folder">
					<Item Name="lvSimController.dll" Type="Document" URL="/&lt;vilib&gt;/rvi/Simulation/lvSimController.dll"/>
				</Item>
			</Item>
			<Item Name="Build Specifications" Type="Build">
				<Item Name="Analog Input" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
					<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
					<Property Name="BuildSpecDecription" Type="Str"></Property>
					<Property Name="BuildSpecName" Type="Str">Analog Input</Property>
					<Property Name="Comp.BitfileName" Type="Str">AnalogInput-FPGA-test.lvbitx</Property>
					<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
					<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
					<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
					<Property Name="Comp.Version.Build" Type="Int">0</Property>
					<Property Name="Comp.Version.Fix" Type="Int">0</Property>
					<Property Name="Comp.Version.Major" Type="Int">1</Property>
					<Property Name="Comp.Version.Minor" Type="Int">0</Property>
					<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
					<Property Name="Comp.Vivado.EnableMultiThreading" Type="Bool">true</Property>
					<Property Name="Comp.Vivado.OptDirective" Type="Str">Default</Property>
					<Property Name="Comp.Vivado.PhysOptDirective" Type="Str">Default</Property>
					<Property Name="Comp.Vivado.PlaceDirective" Type="Str">Default</Property>
					<Property Name="Comp.Vivado.RouteDirective" Type="Str">Default</Property>
					<Property Name="Comp.Vivado.RunPowerOpt" Type="Bool">false</Property>
					<Property Name="Comp.Vivado.Strategy" Type="Str">Default</Property>
					<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
					<Property Name="Comp.Xilinx.MapEffort" Type="Str">default(noTiming)</Property>
					<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
					<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
					<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
					<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
					<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
					<Property Name="DestinationDirectory" Type="Path">/C/Users/UsphereLab/Documents/GitHub/LabView Code/FPGA_Bitfiles</Property>
					<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/C/Users/UsphereLab/Documents/GitHub/LabView Code/FPGA_Bitfiles/AnalogInput-FPGA-test.lvbitx</Property>
					<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path"></Property>
					<Property Name="ProjectPath" Type="Path">/C/Users/UsphereLab/Documents/GitHub/LabView Code/FPGA Testing/Analog Input - R Series.lvproj</Property>
					<Property Name="RelativePath" Type="Bool">false</Property>
					<Property Name="RunWhenLoaded" Type="Bool">true</Property>
					<Property Name="SupportDownload" Type="Bool">true</Property>
					<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
					<Property Name="TargetName" Type="Str">FPGA Target 2</Property>
					<Property Name="TopLevelVI" Type="Ref">/My Computer/FPGA Target 2/Analog Input.vi</Property>
				</Item>
			</Item>
		</Item>
		<Item Name="Analog Input (Host).vi" Type="VI" URL="../Analog Input (Host).vi">
			<Property Name="configString.guid" Type="Str">{008CD651-275E-447A-BA70-735F3B420759}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{038E7F04-5E3B-4603-AC36-C8CE4A6DFC4D}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AI2;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{1041F308-37C6-4C5E-A3EE-94C02F8E0161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{12B90C11-C121-4DFE-8C62-24E51442255D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{14A058DB-8172-4F9F-AFD2-28B18CE983CD}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{221E1121-D533-4CBF-A737-F1BB6712C581}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AI3;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{2B99CF6B-2071-45BA-89FA-CF6599354F78}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AO2;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{2CF5EC02-07DF-4E72-8155-734D002B89E6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{336542BA-C4B3-466E-96B5-412732D2847A}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AI0;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{35828FAE-164E-4516-BF4C-AFF0258177FF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT1;0;ReadMethodType=u8;WriteMethodType=u8{3A9E951A-AE12-4C8A-94F2-C77291F35B75}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AI4;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{409CC936-3F10-4BDC-B254-503FF7C35D42}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AO5;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{45D4AF25-55BB-481D-ADD0-492417651E17}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AO4;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{477DE55D-EF4D-4E36-8937-DB97FB3055D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{4796C5C0-78DF-4F3E-974F-512CC04088B4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{4DFF36D0-0823-417D-BAFA-8182BCA5E678}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AO1;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{5EE331A7-B778-4C67-BFEF-3396249F1546}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{76FB1159-ED97-4482-BB50-59D5F795E124}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{7DB8D115-4F08-44B5-B4E8-AC7D7CFD98E5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{8080AA8A-5AF8-498A-81B5-49BB3B69BC83}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AI6;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{A090ECDA-5632-4CED-BB71-F67DB9C01993}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AO6;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{A5F733FD-1CDE-47D7-BA19-13999A7BCF56}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{AB14BCF9-93C1-449F-91CF-4E51B0A321F9}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AO3;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{B0769666-2416-467B-A85E-A94CC1326A76}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AO7;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{BC5F0CD1-3999-4356-BA19-D2DE22F8D951}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{C01E16FD-B82D-4584-987B-73C1ECB89310}kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AO0;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctl{C0650B72-BF16-4B44-B9FC-9ABDB30A893A}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AI7;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{CB0C98F0-C19C-47ED-AE3A-DFB7FAF5B043}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D310D2A0-2282-4394-A358-415249206D85}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{D9918956-00C6-4FC9-A496-90EED8EF1829}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{DABF13AA-366A-4299-9E2F-4466AAAD7E28}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{E3962CEA-DFFC-41B1-A2D2-B21EEFD72DEE}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AI5;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{EFEC3C41-7BCA-48A2-9C6A-F5D497C4BAA5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{F1F018F9-263A-4ABD-A2B4-E9B04EB77815}kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AI1;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctl{FDC317AD-4D02-4266-A1D4-3F9806FCB423}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT0;0;ReadMethodType=u8;WriteMethodType=u8PXIe-7847R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSPXIE_7847RFPGA_TARGET_FAMILYKINTEX7TARGET_TYPEFPGA/[rSeriesConfig.Begin]rseries.aio./Connector0/AI0=0,rseries.aio./Connector0/AI1=0,rseries.aio./Connector0/AI2=0,rseries.aio./Connector0/AI3=0,rseries.aio./Connector0/AI4=0,rseries.aio./Connector0/AI5=0,rseries.aio./Connector0/AI6=0,rseries.aio./Connector0/AI7=0,rseries.analogCalibratedType=1,rseries.dio./Connector0=0,rseries.dio./Connector1=0,rseries.terminalConfig=0[rSeriesConfig.End]</Property>
			<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;Connector0/AI0kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AI0;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI1kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AI1;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI2kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AI2;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI3kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AI3;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI4kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AI4;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI5kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AI5;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI6kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AI6;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AI7kIntCalDebugEn=0;kIntCalibrateData=0;kIntDelayInClkTicks=38;kIntInitGain=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AI7;0;ReadMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AiFxp.ctlConnector0/AO0kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=0;kIntSimEnable=0;resource=/Connector0/AO0;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO1kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=1;kIntSimEnable=0;resource=/Connector0/AO1;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO2kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=2;kIntSimEnable=0;resource=/Connector0/AO2;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO3kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=3;kIntSimEnable=0;resource=/Connector0/AO3;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO4kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=4;kIntSimEnable=0;resource=/Connector0/AO4;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO5kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=5;kIntSimEnable=0;resource=/Connector0/AO5;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO6kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=6;kIntSimEnable=0;resource=/Connector0/AO6;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/AO7kIntCalDebugEn=0;kIntCalibrateData=0;kIntResChannel=7;kIntSimEnable=0;resource=/Connector0/AO7;0;WriteMethodType=Targets\NI\FPGA\RIO\R Series\78XXR\resource\USB-7855R\Usb7855AoFxp.ctlConnector0/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO0;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO10;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO11;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO12;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO13;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO14;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO15;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO1;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO2;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO3;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO4;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO5;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO6;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO7;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO8;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIO9;0;ReadMethodType=bool;WriteMethodType=boolConnector0/DIOPORT0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT0;0;ReadMethodType=u8;WriteMethodType=u8Connector0/DIOPORT1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Connector0/DIOPORT1;0;ReadMethodType=u8;WriteMethodType=u8PXIe-7847R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSPXIE_7847RFPGA_TARGET_FAMILYKINTEX7TARGET_TYPEFPGA/[rSeriesConfig.Begin]rseries.aio./Connector0/AI0=0,rseries.aio./Connector0/AI1=0,rseries.aio./Connector0/AI2=0,rseries.aio./Connector0/AI3=0,rseries.aio./Connector0/AI4=0,rseries.aio./Connector0/AI5=0,rseries.aio./Connector0/AI6=0,rseries.aio./Connector0/AI7=0,rseries.analogCalibratedType=1,rseries.dio./Connector0=0,rseries.dio./Connector1=0,rseries.terminalConfig=0[rSeriesConfig.End]</Property>
		</Item>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
			</Item>
			<Item Name="NiFpgaLv.dll" Type="Document" URL="NiFpgaLv.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
