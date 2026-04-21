#define MyAppName "EyeFlow"

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define MyAppId MyAppName + "-" + AppVersion
#define MyAppVersionedName MyAppName + " " + AppVersion

#ifndef PayloadDir
  #error PayloadDir must be provided on the ISCC command line.
#endif

#ifndef OutputDir
  #define OutputDir "dist"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppVersionedName}
DefaultDirName={autopf}\{#MyAppName}\{#AppVersion}
DefaultGroupName={#MyAppVersionedName}
DisableProgramGroupPage=yes
LicenseFile={#PayloadDir}\LICENSE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=EyeFlow-setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
SetupIconFile={#PayloadDir}\EyeFlow.ico
UninstallDisplayName={#MyAppVersionedName}
UninstallDisplayIcon={app}\EyeFlow.exe
UsePreviousAppDir=no
UsePreviousGroup=no
UsePreviousTasks=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppVersionedName}"; Filename: "{app}\EyeFlow.exe"
Name: "{autodesktop}\{#MyAppVersionedName}"; Filename: "{app}\EyeFlow.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\EyeFlow.exe"; Description: "Launch {#MyAppVersionedName}"; Flags: nowait postinstall skipifsilent
