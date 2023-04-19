# /share/common/ImageDatasets/imagenet_2012/

# python src/wise_ft.py   \
#     --train-dataset=ImageNet  \
#     --epochs=10  \
#     --lr=0.00003  \
#     --batch-size=512  \
#     --cache-dir=cache  \
#     --model=ViT-B/32  \
#     --eval-datasets=ImageNet \
#     --template=openai_imagenet_template  \
#     --results-db=results.jsonl  \
#     --save=models/wiseft/ViTB32  \
#     --data-location=/share/common/ImageDatasets/imagenet_2012/ \
#     --alpha 1.0

python wise_ft.py   \
    --eval-datasets=ImageNet \
    --load=/share/home/lyq/wise-ft/torchmodel/zeroshot.pt,/share/home/lyq/wise-ft/torchmodel/checkpoint_10.pt  \ 
    --results-db=results.jsonl  \
    --save=/share/home/lyq/wise-ft/myfinetunemodel/  \
    --model=ViT-B/32 \
    --batch-size=256 \
    --data-location=/share/common/ImageDatasets/imagenet_2012/ \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 