diff --git a/README.md b/README.md
index 633977c8658f4564081e3a374046bfc6749abe10..f3818fdf4bf4f84e1894cb2ab73ccccc9199e9f6 100644
--- a/README.md
+++ b/README.md
@@ -1,36 +1,57 @@
 # Awesome-Edge-Detection-Papers
 
 [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
 
 A collection of edge detection papers and corresponding source code/demo program (*a.k.a.* contour detection or boundary detection).
 
 > Feel free to create a PR or an issue. (Pull Request is preferred)
 
 ![examples](https://github.com/MarkMoHR/Awesome-Edge-Detection-Papers/blob/master/edge-detection.png)
 
 
+## Personal Long-Term Memory Agent (Executable Python App)
+
+The repository now includes a runnable Python app that encapsulates the workflow described in the attached task image. The app can ingest multi-modal digital memories, auto-tag them, and surface coherent answers, albums, and summaries.
+
+### Quickstart
+
+```bash
+python memory_agent_app.py ingest "Trip to Shanghai" "Visited the Bund and Yu Garden in Shanghai." "iphone-photos" photo
+python memory_agent_app.py search "Shanghai Bund"
+python memory_agent_app.py qa "When did I visit Shanghai?"
+python memory_agent_app.py timeline --start 2024-01-01
+python memory_agent_app.py album travel
+python memory_agent_app.py summarize --limit 10
+```
+
+Key capabilities include:
+- **Auto-tagging & timestamping:** Automatically generates keyword tags and ISO timestamps when ingesting content.
+- **Retrieval & QA:** Keyword search, time-filtered timelines, tag-based albums, and simple factoid answers using stored context.
+- **Content generation:** Produces short summaries or album narratives from relevant memories.
+
+
 **Outline**
 
 - [Edge detection related dataset](#0-edge-detection-related-dataset)
 - [Deep-learning based approaches](#1-deep-learning-based-approaches)
   - [General edge detection](#11-general-edge-detection)
   - [Object contour detection](#12-object-contour-detection)
   - [Semantic edge detection (Category-Aware)](#13-semantic-edge-detection-category-aware)
   - [Occlusion boundary detection](#14-occlusion-boundary-detection)
   - [Edge detection from multi-frames](#15-edge-detection-from-multi-frames)
 - [Traditional approaches](#2-traditional-approaches)
 - [[Misc] Useful Links](#3-useful-links)
 
 
 ## 0. Edge detection related dataset
 
 | Short name | Source Paper | Source | Introduction |
 | --- | --- | --- | --- |
 | [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) | [Contour Detection and Hierarchical Image Segmentation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) | TPAMI 2011 | Classical edge detaction dataset. |
 | [NYUDv2](https://github.com/s-gupta/rgbd#notes) | [Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Gupta_Perceptual_Organization_and_2013_CVPR_paper.pdf) | CVPR 2013 | Edges come from the boundary of annotated segmentation mask. |
 | [Multi-cue](https://serre-lab.clps.brown.edu/resource/multicue/) | [A systematic comparison between visual cues for boundary detection](https://pubmed.ncbi.nlm.nih.gov/26748113/) | Vision research 2016 | With boundary annotations. |
 | [Wireframe](https://github.com/cherubicxn/hawp#data-preparation) | [Holistically-Attracted Wireframe Parsing](https://arxiv.org/pdf/2003.01663) | CVPR 2020 | Edges come from the annotated wireframe. |
 
 ---
 
 ## 1. Deep-learning based approaches
