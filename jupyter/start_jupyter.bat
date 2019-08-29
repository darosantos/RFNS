@echo off
cls

cd %cd%

date /t

echo Computador: %computername%        Usuario: %username%

echo 'Aguarde enquanto iniciamos o notebook jupyter'

jupyter notebook 

echo 'Pronto!'
pause