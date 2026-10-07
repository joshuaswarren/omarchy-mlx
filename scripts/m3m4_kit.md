# m3m4_kit — why each step exists

`bash scripts/m3m4_kit.sh` on an M3 or M4 Mac. Nothing is uploaded
without a second explicit `--submit` command; nothing needs root.

We have no M3 or M4 silicon in the lab. Static facts (ADT dumps, IPSW
firmware lists, aurora device trees) are already extracted and cited in
omarchy-ane `data/ane-soc/`. The questions below are the ones static
analysis cannot answer — each kit step exists to answer one of them.

## Linux (aurora 12.3), M3

| Step | Hardware-only question it answers |
|---|---|
| aurora `--m3-report` | Which devices does the kernel actually bind on this board (t8122/t6030/t6031/t6034), what is missing, and what does the boot loader hand over? Every board bins differently; only the running kernel answers. |
| aurora `--m3-power-survey` (optional, sudo) | What power/thermal envelope does this chip run under Linux? No ADT field predicts it. |
| collector `--ane-smoke` path | Does the ANE bind and pass the H13/H14 golden smoke on H15 silicon? The ane/h15 module's stage 0/1 read registers fine, but only silicon proves the mailbox wakes and returns RTKit HELLO (docs/h15-volunteer.md). |
| collector correctness sections | Do MLX ops on the M3 GPU (G15G, aurora `mesa-m3`) return exactly what the pure-python reference returns? Mesa maturity on G15G is the open risk; static code review cannot grade it. |
| collector benchmark sections | What decode/prefill numbers does the M3 GPU reach, to compare against the macOS MLX numbers from the same box? |

## macOS, M3/M4 (M4 is macOS-only today)

| Step | Hardware-only question it answers |
|---|---|
| collector (native) | What are the reference MLX/Metal numbers and ANE facts on the same silicon? This is the baseline every Linux number is judged against. |
| collector ANE dump | What ANE version/cores/firmware does macOS expose on this chip (H15 `iop,ascwrap-v6`, H16 `ane,t8020`)? Confirms the Linux driver targets the right generation. |
| powermetrics one-shot (optional, sudo) | Which GPU/ANE power channels exist and what do they read at idle? The channels an M4 Linux driver would need to expose. |
| `MKIT_HWX=1` compile (optional) | Does the mil-hwx-compiler's H16G HWX output build on a stock macOS toolchain? Compiler evidence that H16 objects are well-formed; runtime acceptance still needs silicon. |

## After the kit

Send the collector archive with the printed `--submit` command (it
answers with a public receipt URL) and attach the aurora tgz. A
maintainer reads the row and, if it passes the promotion rules
(omarchy-ane README), the chip's status changes — one clean row is
enough by design.
