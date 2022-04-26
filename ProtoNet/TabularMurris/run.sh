# 1-shot
python main.py --datasource=TM --metatrain_iterations=5000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --mix=1
python main.py --datasource=TM --metatrain_iterations=5000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --mix=1 --train=0 --test_epoch=xxx

# 5-shot
python main.py --datasource=TM --metatrain_iterations=2000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --mix=1
python main.py --datasource=TM --metatrain_iterations=2000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --mix=1 --train=0 --test_epoch=xxx