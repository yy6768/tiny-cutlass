@echo off
rem Compatibility entry; the family workflow is maintained in swin.bat.
call "%~dp0swin.bat" %*
exit /b %errorlevel%
