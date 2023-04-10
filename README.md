# Segment Anything Demo

> **[Meta AI Research, FAIR](https://ai.facebook.com/research/)**
>
> [Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com/), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)
>
> [[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](https://github.com/facebookresearch/segment-anything#citing-segment-anything)]

## 运行结果

<a href="https://github.com/laorange/segment-anything-demo/blob/master/main.ipynb">在线预览 Jupyter Notebook</a>
  
## 本地运行

1. 下载代码；

2. 下载 `.pth` 文件，并放置于 `./model/` 文件夹下。

   - [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)；

   - 若要使用其他模型，可前往 [model-checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) 下载，并修改[模型名称](https://github.com/laorange/segment-anything-demo/blob/f22d2e37badbafbde8d885371f6bf360f3eeea73/main.py#L40-L41)。

3. 运行程序



