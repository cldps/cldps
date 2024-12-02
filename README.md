Diffusion models (DMs) have recently shown great promise in solving inverse problems. While most research in this area addresses non-blind inverse problems, where the measurement operator is assumed to be known, real-world applications frequently involve blind inverse problems with unknown measurements. Existing DM-based methods for blind inverse problems are limited, primarily addressing only linear measurements and thus lacking applicability to real-life scenarios that often involve non-linear operations. To overcome these limitations, we propose CL-DPS, a novel approach based on contrastive learning for solving blind inverse problems via diffusion posterior sampling. In CL- DPS, we first train an auxiliary deep neural network (DNN) offline using a modified version of MoCo [23], a contrastive learning technique. This auxiliary DNN serves as a likelihood estimator, enabling the estimation of p(y|x) without prior knowledge of the measurement operator, thereby adjusting the reverse path of the diffusion process for inverse problem-solving. Additionally, we introduce an overlapped patch-wise inference method to improve the accuracy of likelihood estimation. Extensive qualitative and quantitative experiments demonstrate that CL-DPS effectively addresses non-linear inverse problems, such as rotational de-blurring, which previous methods could not solve.

<img width="953" alt="image" src="https://github.com/user-attachments/assets/82610919-fc08-4e64-a46c-2231c5c5680e">