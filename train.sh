# NI-UDA for Office-Home S65 to S25 to T25
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Art_small_25.txt --tgt_address  ../data/officehome/Clipart_small_25.txt --src_ns_address ../data/officehome/Art_noshare_40.txt --gpu_id 0
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Art_small_25.txt --tgt_address  ../data/officehome/Product_small_25.txt --src_ns_address ../data/officehome/Art_noshare_40.txt --gpu_id 0
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Art_small_25.txt --tgt_address  ../data/officehome/Real_World_small_25.txt --src_ns_address ../data/officehome/Art_noshare_40.txt --gpu_id 0
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Clipart_small_25.txt --tgt_address  ../data/officehome/Art_small_25.txt --src_ns_address ../data/officehome/Clipart_noshare_40.txt --gpu_id 0
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Product_small_25.txt --tgt_address  ../data/officehome/Art_small_25.txt --src_ns_address ../data/officehome/Product_noshare_40.txt --gpu_id 0
python train.py --config ../config/officehome_NIUDA.yml --src_address  ../data/officehome/Real_World_small_25.txt --tgt_address  ../data/officehome/Art_small_25.txt --src_ns_address ../data/officehome/Real_World_noshare_40.txt --gpu_id 0
# UDA for Office-Home S65 to T65
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Art_65.txt --tgt_address  ../data/officehome/Clipart_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Art_65.txt --tgt_address  ../data/officehome/Product_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Art_65.txt --tgt_address  ../data/officehome/Real_World_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Clipart_65.txt --tgt_address  ../data/officehome/Art_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Clipart_65.txt --tgt_address  ../data/officehome/Product_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Clipart_65.txt --tgt_address  ../data/officehome/Real_World_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Product_65.txt --tgt_address  ../data/officehome/Art_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Product_65.txt --tgt_address  ../data/officehome/Clipart_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Product_65.txt --tgt_address  ../data/officehome/Real_World_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Real_World_65.txt --tgt_address  ../data/officehome/Art_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Real_World_65.txt --tgt_address  ../data/officehome/Clipart_65.txt --gpu_id 0
python train.py --config ../config/officehome_UDA.yml --src_address  ../data/officehome/Real_World_65.txt --tgt_address  ../data/officehome/Product_65.txt --gpu_id 0
# NI-UDA for DomainNet S345 to S50 to T50
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/infograph_50.txt --tgt_address  ../data/domainnet/painting_50.txt --src_ns_address ../data/domainnet/infograph_noshare_295.txt --gpu_id 0
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/infograph_50.txt --tgt_address  ../data/domainnet/sketch_50.txt --src_ns_address ../data/domainnet/infograph_noshare_295.txt --gpu_id 0
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/infograph_50.txt --tgt_address  ../data/domainnet/real_50.txt --src_ns_address ../data/domainnet/infograph_noshare_295.txt --gpu_id 0
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/painting_50.txt --tgt_address  ../data/domainnet/infograph_50.txt --src_ns_address ../data/domainnet/painting_noshare_295.txt --gpu_id 0
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/sketch_50.txt --tgt_address  ../data/domainnet/infograph_50.txt --src_ns_address ../data/domainnet/sketch_noshare_295.txt --gpu_id 0
python train.py --config ../config/domainnet_NIUDA.yml --src_address  ../data/domainnet/real_50.txt --tgt_address  ../data/domainnet/infograph_50.txt --src_ns_address ../data/domainnet/real_noshare_295.txt --gpu_id 0
