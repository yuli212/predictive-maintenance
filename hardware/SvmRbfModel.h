#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float kernels[11] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   -0.890162033108  , -0.849624306651 );
                        kernels[1] = compute_kernel(x,   -0.802718490979  , -0.910411635392 );
                        kernels[2] = compute_kernel(x,   -0.960980859851  , -1.096903755547 );
                        kernels[3] = compute_kernel(x,   0.535418025898  , 0.086084757484 );
                        kernels[4] = compute_kernel(x,   1.914976576545  , 0.648783873573 );
                        kernels[5] = compute_kernel(x,   -0.078838362326  , 1.412339723834 );
                        kernels[6] = compute_kernel(x,   0.615786716103  , 1.841138650991 );
                        kernels[7] = compute_kernel(x,   1.589687274526  , 2.094026633638 );
                        kernels[8] = compute_kernel(x,   2.021311454346  , 1.800936999354 );
                        kernels[9] = compute_kernel(x,   -0.7699012398  , 0.530747728757 );
                        kernels[10] = compute_kernel(x,   1.627921799738  , 0.048759283925 );
                        float decision = 0.627137116438;
                        decision = decision - ( + kernels[0] * -1.178589435259  + kernels[1] * -0.096572770813  + kernels[2] * -0.476579986661 );
                        decision = decision - ( + kernels[3] * 0.288643975352  + kernels[4] * 0.149294645539  + kernels[5] * 0.060949678254  + kernels[6] * 0.241207063847  + kernels[7] * 0.095439080196  + kernels[8] * 0.21852780644  + kernels[9] * 0.522741044438  + kernels[10] * 0.174938898668 );

                        return decision > 0 ? 0 : 1;
                    }

                    /**
                    * Predict readable class name
                    */
                    const char* predictLabel(float *x) {
                        return idxToLabel(predict(x));
                    }

                    /**
                    * Convert class idx to readable name
                    */
                    const char* idxToLabel(uint8_t classIdx) {
                        switch (classIdx) {
                            case 0:
                            return "Normal";
                            case 1:
                            return "Indikasi_Kotor";
                            default:
                            return "Houston we have a problem";
                        }
                    }

                protected:
                    /**
                    * Compute kernel between feature vector and support vector.
                    * Kernel type: rbf
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, 2);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 2; i++) {
                            kernel += pow(x[i] - va_arg(w, double), 2);
                        }

                        return exp(-1.0 * kernel);
                    }
                };
            }
        }
    }