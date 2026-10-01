cd /d "%~dp0.."
if not exist bin mkdir bin
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cl /nologo /O2 /Fobin\test.obj /Febin\test.exe samples\test.c
bin\test.exe
