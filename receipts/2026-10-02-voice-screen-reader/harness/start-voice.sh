#!/bin/bash
# Container bootstrap for the voice screen-reader pass.
#
# Drives: dbus -> Xvfb :99 -> speech-dispatcher (sd_dummy debug) -> Orca
# user-settings -> fake-mic WAV generation -> stub backend. Chromium is
# started separately by the driver so it can own the fake-media flags
# and relaunch with different capture WAVs per scenario. No host paths
# are referenced.
set -e
mkdir -p /srv/logs /srv/out
cd /srv

# 1. dbus session
eval "$(dbus-launch --sh-syntax)"
export DBUS_SESSION_BUS_ADDRESS
echo "$DBUS_SESSION_BUS_ADDRESS" > /srv/logs/dbus.env

# 2. Xvfb
Xvfb :99 -screen 0 1440x900x24 -nolisten tcp > /srv/logs/xvfb.log 2>&1 &
sleep 1
export DISPLAY=:99

# 3. speech-dispatcher (sd_dummy debug + logging)
export XDG_RUNTIME_DIR=/tmp/sr-xdg
mkdir -p "$XDG_RUNTIME_DIR/speech-dispatcher"
chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p /srv/logs/speech-dispatcher/modules
cat > /srv/logs/speech-dispatcher/speechd.conf <<'EOF'
DefaultModule sd_dummy
LogLevel 3
AddModule "dummy" "sd_dummy" ""
EOF
cat > /srv/logs/speech-dispatcher/modules/sd_dummy.conf <<'EOF'
DebugLogFile /srv/logs/speechd-utterances.log
EOF
speech-dispatcher -d > /srv/logs/speechd-startup.log 2>&1 &
sleep 2

# 4. stub backend
python3 /srv/harness/stub_server.py > /srv/logs/stub.log 2>&1 &
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:8765/api/session > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# 5. fake microphone WAVs
python3 - <<'EOF'
import math, struct, wave
for name, amp, gated in (("fake-mic-tone.wav", 0.25, True),
                         ("fake-mic-silence.wav", 0.002, False)):
    with wave.open(f"/srv/logs/{name}", "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        frames = bytearray()
        for i in range(16000 * 60):
            if gated and (i % 16000) >= 9600:
                frames += struct.pack("<h", 0)
                continue
            v = int(max(-1.0, min(1.0,
                amp * math.sin(2 * math.pi * 440.0 * i / 16000))) * 32767)
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
EOF

# 6. minimal Orca settings (no Learn Mode, speech on)
mkdir -p /tmp/.config/orca
cat > /tmp/.config/orca/orca.user-settings.py <<'EOF'
import orca.settings
orca.settings.settingsDict = {
    'enableSpeech': True,
    'enableBraille': False,
    'profile': 'default',
}
EOF
orca --debug-file=/srv/logs/orca-debug.log \
     --disable-splash --no-setup > /srv/logs/orca-stderr.log 2>&1 &
sleep 4

echo "container ready; run: python3 /srv/harness/voice-driver.py all"

# Keep the container alive until the driver finishes; on exit kill the
# background processes this script owns. Orba
trap "pkill -f Xvfb; pkill -f speech-dispatcher; pkill -f '^python3 /srv/harness/stub_server'; pkill -f 'orca --debug-file' 2>/dev/null || true" EXIT
wait