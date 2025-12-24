# Thumb-Print-Matching
Thumb Print Matching

<p align="center">
  <img src="/assets/Image%208-5-25%20at%202.40%E2%80%AFPM.JPG" width="240" />
  <img src="/assets/Image%208-5-25%20at%202.43%E2%80%AFPM.JPG" width="240" />
  <img src="/assets/Image%208-5-25%20at%202.44%E2%80%AFPM.JPG" width="240" />
  <br/>
  <em>Thumbprints Labeled by Minutiae from Left to Right: (Whorls, Loops, and Deltas), (Ridge Orientation), and (Terminations and Bifurcations)</em>
</p>


Current fingerprint matching strategy:
When first given a print, we have to clean the image of spurious noise. After the image is cleaned, we need to classify unique features that will help us identify the fingerprint owner.

We first segment the image.
Within each segment of the image, we conduct a least squares regression analysis to determine the slope of the ridge [1]. This results in a matrix of orientations that represents the slopes of the ridges at every instance in the image. This can be visualized as a vector field. Three unique features can be derived from the orientation matrix: whorls, loops, and deltas. These are classified by parsing the entirety of the matrix. At every index, we explore the neighboring segments. If the sum of the difference of the neighboring segments is -180 degrees, 360 degrees, or 180 degrees, the region is classified as either an arch, whorl, or loop, respectively [1].
To extract more feature types from the image, we can segment the image again to find ridge endings and ridge bifurcations (minutiae) [2].

Once we have identified and classified all the features, we need a systematic way to determine if the features correspond to the owner. There is a lot of discourse discussing the best strategy to go about this. We can’t assume the orientation of the print that is captured. So, how do we combat this when we want to conduct authentication? In 2003, M. Tico and P. Kuosmanen introduced the strategy of rotating the collected image on top of the stored print [3]. At every angle, a heuristic is calculated by tallying the distance between key features. If the heuristic surpasses a predetermined threshold, enough features overlaped, the finger print’s owner is verified [3]. After speaking with a member on the TouchID algorithms team at Apple, I found that Apple employs this strategy by comparing the collected image segment, parsing the stored image, and at every index, rotating the image to see if features align.

When meeting with a member from the biometric computer vision team at Apple, I asked about tradeoffs in classifying prints by considering respective relationships between featurues localy as opposed to globaly.

In 2024, A. Liu et al. proposed that utilizing a hybrid graph transformer neural network could improve classification accuracy when confronted with noisy prints [4]. Liu demonstrates that mapping relationships between neighboring features can aid in constructing a graph that establishes global relationships between features [4]. In other words, unique features will be cast as nodes, and edges will span between the three nearest neighbors [4]. This strategy allows us to forgo the orientation issue introduced before. However, there are some issues with the transformer-based model. A labeled dataset is required to train the model to learn which node relationships to weigh heavily [4]. Although employing a Graph Attention mechanism to capture important relationships between nodes makes the model less susceptible to noise, this makes it harder to compare a subgraph [4]. Additionally, when constructing the nodes of the graph, the model does not consider a difference between various feature classifications, but rather their coordinates [4]. This can pose an issue for several reasons: when comparing against a rotated and/or segmented print, the same features will be labeled differently.

New fingerprint matching proposal:
I propose a new strategy, keeping the same graph structure discussed by A. Liu et al. [4]; however, rather than physical proximity, edges span between nodes if they lie on the same integral curve of the vector field that represents ridge orientations. This allows us to construct a graph that's derived from a key unique characteristic of every fingerprint: ridge orientations [1].

Redefining the input print as a graph allows us to neglect the reorientation concern. Instead of using a graph neural network to classify the image, we should consider the Weisfeiler-Leman (WL) algorithm to determine if two graphs (input print and true print) are isomorphic [5]. Additionally, the Weisfeiler-Leman procedure refines node labels based on neighboring nodes' labels, which allows use to consider feature's relationships globally and locally [5]. Related work also generalizes Weisfeiler–Lehman kernels to subgraph settings, allowing us to authenticate even if only a partial print is captured[6]. This proposed strategy allows us to consider many different feature types when determining authenticity, including ridge orientations, loops, whorls, deltas, bifurcations, and terminations [2], [5].

<p align="center">
  <img src=/assets/Image%208-19-25%20at%2012.49%E2%80%AFPM.JPG width="520" />
  <br/>
  <em>Probability Distribution of the Isomorphic scores for true pairs (blue) and false pairs (red).</em>
</p>
Having implemented the algorithm, I ran a series of tests to determine its accuracy. When given two graphs, the algorithm will return the probability that the two graphs are truly isomorphic (the print belongs to the owner). I iterated over a dataset of prints. I cleaned, extracted features, and developed graphs for image [2]. I simulated user input by reconstructing the graph for a rotated version of the image. I then passed to the Weisfeiler-Leman algorithm every possible pair that can be composed from the dataset [5]. Rotation was determined randomly. Across a dataset of 320, therefore {320 choose 2} pairs. I achieved an AUROC score of .97, demonstrating the algorithm's robust interclass discrimination (ability to rate true pairs higher than false pairs 93% of the time). I found that there was no correlation between the degree of rotation and classification accuracy. Thereby proving that the algorithm is not susceptible to user orientation input and has a high authentication accuracy.


#Bibliography

[1] Lin, Hong, et al., Fingerprint Image Enhancement: Algorithm and Performance Evaluation (1998).

[2] Ł. Więcław, A Minutiae-Based Matching Algorithms in Fingerprint Recognition Systems (2009).

[3] Tico, M., and P. Kuosmanen, Fingerprint Matching Using an Orientation-Based Minutia Descriptor (2003).

[4] Liu, Aitong, et al., A Hybrid Graph Transformer Network for Fingerprint and Palmprint Minutiae Matching of Young Children (2025).

[5] Morris, Christopher, et al., The Power of the Weisfeiler-Leman Algorithm for Machine Learning with Graphs (2021).

[6] Kim, Dongkwan, and Alice Oh, Generalizing Weisfeiler-Lehman Kernels to Subgraphs (2022).


(put db3 in a path /fingerprint/db3)
