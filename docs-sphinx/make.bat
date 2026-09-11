@ECHO OFF
pushd %~dp0
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build
if "%1" == "" goto help
if "%1" == "html" goto html
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end
:html
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
xcopy /E /I /Y %BUILDDIR%\html ..\docs
goto end
:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
:end
popd
