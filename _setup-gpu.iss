#define MyAppName "Rembg GPU"
#define MyAppVersion "STABLE"
#define MyAppPublisher "danielgatis"
#define MyAppURL "https://github.com/danielgatis/rembg"
#define MyAppExeName "rembg.exe"
#define MyAppId "49AB7484-212F-4B31-A49F-533A480F3FD4"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Rembg
DefaultGroupName=Rembg
DisableProgramGroupPage=yes
OutputBaseFilename=rembg-cli-gpu-installer
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir=dist
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourcePath}dist\rembg-gpu\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}dist\rembg-gpu\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: modifypath; Description: "Add to PATH variable"

[Icons]
Name: "{group}\Rembg"; Filename: "{app}\{#MyAppExeName}"

[Code]
const
    ModPathName = 'modifypath';
    ModPathType = 'user';

function ModPathDir(): TArrayOfString;
begin
    setArrayLength(Result, 1)
    Result[0] := ExpandConstant('{app}');
end;
#include "_modpath.iss"
