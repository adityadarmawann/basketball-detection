# Audit & Fix Plan — Smart Vision Basketball Analytics (BBOX/Stats not appearing)

## Step 1 — Confirm pipeline trigger
- Check backend logs on POST `/api/upload-video`:
  - apakah ada warning `VideoProcessor unavailable — processing will be skipped`
  - apakah background task `_run_pipeline` masuk dan memulai `VideoProcessor.process_video`

## Step 2 — Confirm WS real connection
- In browser DevTools → Network → WS:
  - apakah connect ke `ws://.../api/ws/live?match_id=...` benar-benar `Open`
  - apakah jatuh ke fallback mock setelah 3 detik

## Step 3 — Confirm Redis/WS broadcast path
- Pastikan salah satu jalan:
  - A) `ws_manager` tidak `None` dan broadcast via `ws_manager.broadcast`
  - B) Redis tersedia + WS live WS handler mendengar channel `match:{match_id}`
- Add/cek logging saat publish `match:{match_id}` dan saat broadcast.

## Step 4 — Validate tracker produced bboxes
- Tambahkan logging ringkas per frame:
  - `len(tracked_players)`
  - contoh `player.get('bbox')`
- Kalau `tracked_players` kosong atau bbox kosong => fokus ke model detector/tracker load atau stride.

## Step 5 — Validate frontend mapping
- Pastikan frontend membaca `frame_update` message dan komponen overlay memakai `player.bbox`.
- Pastikan tidak ada mismatch schema `box` normalized vs expected.

## Step 6 — Implement minimal fix untuk “progress 1 no output”
- Patch terkecil untuk membuat minimal `frame_update` sampai ke UI dengan data dummy if needed:
  - jika WS real connect gagal, jangan langsung mock tanpa menampilkan error.
  - atau kirim heartbeat frame_update dari backend walau bbox kosong.

## Step 7 — Testing
- Upload 1 video test.
- Verify: bbox overlay muncul, scoreboard jalan, dan stats minimal terisi.

