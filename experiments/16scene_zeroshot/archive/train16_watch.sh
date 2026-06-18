cd /home/hanyan_arch/jaehyeon/WRF-GSplus
while ! grep -q "\[it  12000\]\|\[it 12000\]" /tmp/train16.log 2>/dev/null; do sleep 60; done
sleep 5
echo "=== 16-SCENE TRAINING DONE ==="
echo "--- 학습 곡선 (zero-shot DoA per eval) ---"
grep -E "ZERO-SHOT" /tmp/train16.log
echo "--- 최종 TRAIN-scene (마지막) ---"
grep -E "TRAIN-scene" /tmp/train16.log | tail -1
