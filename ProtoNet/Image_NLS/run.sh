# 1-shot
# miniImagenet-S
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=xxx

# ISIC
python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=2 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1
python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=2 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --train=0 --test_epoch=xxx

# DermNet-S
python main.py --datasource=dermnet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1
python main.py --datasource=dermnet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --train=0 --test_epoch=xxx

# 5-shot
# miniImagenet-S
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=xxx

# ISIC
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --train=0 --test_epoch=xxx

# DermNet-S
python main.py --datasource=dermnet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2
python main.py --datasource=dermnet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=xxx