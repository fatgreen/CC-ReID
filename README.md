# CC-ReID

Cloth-Changing Re-IDentification

For demo_single_image.py
python D:/AIM-CCReID-main/demo_single_image.py --cfg configs/res50_cels_cal.yaml --img_path D:\AIM-CCReID-main\prcc\rgb\test\A\018 --weights D:\AIM-CCReID-main\OUTPUT_PATH\prcc\eval\best_model.pth.tar --gpu 0


python main.py --cfg configs/res16_cels_cal.yaml configs/res50_cels_cal.yaml configs/res101_cels_cal.yaml --weights D:/AIM-CCReID-main/prcc.pth.tar --gpu 0 --dataset prcc

python D:/AIM-CCReID-main/demo_single_image.py --cfg configs/res50_cels_cal.yaml --img_folder D:\AIM-CCReID-main\prcc\rgb\test\A\059 --weights D:/AIM-CCReID-main/prcc.pth.tar --gpu 0

python D:/AIM-CCReID-main/demo_single_image.py --cfg configs/res101_cels_cal.yaml --img_folder D:\AIM-CCReID-main\prcc\rgb\test\A\059 --weights D:\AIM-CCReID-main\OUTPUT_PATH\prcc\eval\best_model.pth.tar --gpu 0
