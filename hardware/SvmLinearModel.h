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
                        float kernels[4] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   -0.890162033108  , -0.849624306651 );
                        kernels[1] = compute_kernel(x,   -0.860095834845  , -0.86539992482 );
                        kernels[2] = compute_kernel(x,   0.535418025898  , 0.086084757484 );
                        kernels[3] = compute_kernel(x,   -0.7699012398  , 0.530747728757 );
                        float decision = 0.623607956353;
                        decision = decision - ( + kernels[0] * -1.0  + kernels[1] * -0.100167118252 );
                        decision = decision - ( + kernels[2] * 0.266583509962  + kernels[3] * 0.83358360829 );

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
                    * Kernel type: linear
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, 2);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 2; i++) {
                            kernel += x[i] * va_arg(w, double);
                        }

                        return kernel;
                    }
                };
            }
        }
    }