@echo off
title Aria - Sinkronisasi Satelit Sentinel-5P (Loop 4 Jam)
cd /d "%~dp0"

:: Deteksi runner yang tersedia (uv atau python venv)
set "RUNNER=uv"
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "RUNNER=%USERPROFILE%\.local\bin\uv.exe"
    ) else if exist ".\.venv\Scripts\python.exe" (
        set "RUNNER=.\.venv\Scripts\python.exe"
    )
)

:LOOP
cls
echo =====================================================================
echo  ARIA - SINKRONISASI SATELIT SENTINEL-5P (LOOP SETIAP 4 JAM)
echo  Waktu Mulai : %DATE% %TIME%
echo  Wilayah     : Seluruh DKI Jakarta (5 Wilayah)
echo  Rentang     : Otomatis H-8 s/d Hari Ini
echo =====================================================================
echo.

if "%RUNNER%"==".\.venv\Scripts\python.exe" (
    "%RUNNER%" .\commands\seed_satellite_data.py
) else (
    "%RUNNER%" run .\commands\seed_satellite_data.py
)

echo.
echo =====================================================================
echo  Sinkronisasi putaran ini selesai!
echo  Jadwal berikutnya: 4 jam lagi (14.400 detik).
echo  - Tekan sembarang tombol jika ingin sinkronisasi SEKARANG (tanpa tunggu).
echo  - Tutup jendela CMD ini (atau tekan Ctrl+C) jika ingin menghentikan loop.
echo =====================================================================
echo.

:: Countdown 4 jam (14.400 detik). Jika user menekan tombol apa saja, langsung lanjut siklus berikutnya
timeout /t 14400

goto LOOP
