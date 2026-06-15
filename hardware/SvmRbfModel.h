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
                        float kernels[60] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   -0.026891486984  , 0.095692707678 );
                        kernels[1] = compute_kernel(x,   0.04912685078  , -0.174816713319 );
                        kernels[2] = compute_kernel(x,   -0.099454445758  , 0.353906245902 );
                        kernels[3] = compute_kernel(x,   -0.061445276876  , 0.218651535404 );
                        kernels[4] = compute_kernel(x,   -0.095999066769  , 0.34161036313 );
                        kernels[5] = compute_kernel(x,   0.011117681898  , -0.039562002821 );
                        kernels[6] = compute_kernel(x,   0.038760713812  , -0.137929065002 );
                        kernels[7] = compute_kernel(x,   -0.08908830879  , 0.317018597584 );
                        kernels[8] = compute_kernel(x,   0.035305334823  , -0.125633182229 );
                        kernels[9] = compute_kernel(x,   -0.071811413844  , 0.255539183721 );
                        kernels[10] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[11] = compute_kernel(x,   -0.085632929801  , 0.304722714812 );
                        kernels[12] = compute_kernel(x,   0.021483818866  , -0.076449651139 );
                        kernels[13] = compute_kernel(x,   0.145877462479  , -0.519101430952 );
                        kernels[14] = compute_kernel(x,   -0.078722171823  , 0.280130949267 );
                        kernels[15] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[16] = compute_kernel(x,   0.028394576844  , -0.101041416684 );
                        kernels[17] = compute_kernel(x,   0.024939197855  , -0.088745533911 );
                        kernels[18] = compute_kernel(x,   -0.082177550812  , 0.292426832039 );
                        kernels[19] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[20] = compute_kernel(x,   -0.075266792833  , 0.267835066494 );
                        kernels[21] = compute_kernel(x,   0.031849955833  , -0.113337299456 );
                        kernels[22] = compute_kernel(x,   0.145877462479  , -0.519101430952 );
                        kernels[23] = compute_kernel(x,   0.018028439876  , -0.064153768366 );
                        kernels[24] = compute_kernel(x,   -0.08908830879  , 0.317018597584 );
                        kernels[25] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[26] = compute_kernel(x,   -0.068356034855  , 0.243243300949 );
                        kernels[27] = compute_kernel(x,   0.145877462479  , -0.519101430952 );
                        kernels[28] = compute_kernel(x,   0.014573060887  , -0.051857885593 );
                        kernels[29] = compute_kernel(x,   -0.09254368778  , 0.329314480357 );
                        kernels[30] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[31] = compute_kernel(x,   -0.064900655866  , 0.230947418176 );
                        kernels[32] = compute_kernel(x,   0.042216092801  , -0.150224947774 );
                        kernels[33] = compute_kernel(x,   0.145877462479  , -0.519101430952 );
                        kernels[34] = compute_kernel(x,   0.007662302909  , -0.027266120048 );
                        kernels[35] = compute_kernel(x,   -0.099454445758  , 0.353906245902 );
                        kernels[36] = compute_kernel(x,   -0.199660436447  , 0.710486846307 );
                        kernels[37] = compute_kernel(x,   -0.057989897887  , 0.206355652631 );
                        kernels[38] = compute_kernel(x,   0.04567147179  , -0.162520830547 );
                        kernels[39] = compute_kernel(x,   0.145877462479  , -0.519101430952 );
                        kernels[40] = compute_kernel(x,   -0.102909824747  , 0.366202128675 );
                        kernels[41] = compute_kernel(x,   -0.054534518898  , 0.194059769858 );
                        kernels[42] = compute_kernel(x,   -0.026891486984  , 0.095692707678 );
                        kernels[43] = compute_kernel(x,   0.049930177533  , -0.27032515663 );
                        kernels[44] = compute_kernel(x,   0.205664304026  , -0.453901990928 );
                        kernels[45] = compute_kernel(x,   0.045429399813  , -0.532258721015 );
                        kernels[46] = compute_kernel(x,   0.193691513553  , -0.22599745599 );
                        kernels[47] = compute_kernel(x,   -0.177563580982  , 0.075956772583 );
                        kernels[48] = compute_kernel(x,   0.053066373723  , 0.552363184845 );
                        kernels[49] = compute_kernel(x,   -0.403208614014  , 0.600959705643 );
                        kernels[50] = compute_kernel(x,   0.048004341228  , 0.755675954238 );
                        kernels[51] = compute_kernel(x,   -0.39123582354  , 0.373055170705 );
                        kernels[52] = compute_kernel(x,   0.349425640046  , -0.409574290288 );
                        kernels[53] = compute_kernel(x,   -0.106849347691  , -0.36097776949 );
                        kernels[54] = compute_kernel(x,   0.123780607014  , 0.115428642773 );
                        kernels[55] = compute_kernel(x,   -0.24747448752  , 0.417382871345 );
                        kernels[56] = compute_kernel(x,   -0.099212373781  , 0.723644136371 );
                        kernels[57] = compute_kernel(x,   -0.259447277994  , 0.645287406283 );
                        kernels[58] = compute_kernel(x,   -0.103713151501  , 0.461710571985 );
                        kernels[59] = compute_kernel(x,   -0.026891486984  , 0.095692707678 );
                        float decision = 6.760843920207;
                        decision = decision - ( + kernels[0] * -7.705811422045  + kernels[1] * -5.425658664121  + kernels[2] * -36.029888111987  + kernels[3] * -1.153623495178  + kernels[4] * -23.684591904857  + kernels[5] * -0.065519023937  + kernels[6] * -32.216881640821  + kernels[7] * -2.740804450223  + kernels[8] * -10.603931950642  + kernels[9] * -0.249353933672  + kernels[10] * -0.62139572457  + kernels[11] * -22.687384323603  + kernels[12] * -0.347935880832  + kernels[13] * -3.402575559931  + kernels[14] * -0.617767171179  + kernels[15] * -1.341056279358  + kernels[16] * -0.190906378943  + kernels[17] * -27.362577998143  + kernels[18] * -0.484639797417  + kernels[19] * -43.516662223471  + kernels[20] * -10.711548838065  + kernels[21] * -50.0  + kernels[22] * -7.092505399456  + kernels[23] * -0.267948840269  + kernels[24] * -0.085000037531  + kernels[25] * -49.450210973461  + kernels[26] * -2.382324308065  + kernels[27] * -50.0  + kernels[28] * -44.860561446222  + kernels[29] * -38.672127236081  + kernels[30] * -50.0  + kernels[31] * -0.760245738567  + kernels[32] * -45.26364288414  + kernels[33] * -49.874951238455  + kernels[34] * -0.070431399179  + kernels[35] * -50.0  + kernels[36] * -49.994751093516  + kernels[37] * -1.002477622286  + kernels[38] * -11.459292482218  + kernels[39] * -50.0  + kernels[40] * -28.566864623271  + kernels[41] * -9.688164788324 );
                        decision = decision - ( + kernels[42] * 50.0  + kernels[43] * 50.0  + kernels[44] * 50.0  + kernels[45] * 50.0  + kernels[46] * 50.0  + kernels[47] * 50.0  + kernels[48] * 50.0  + kernels[49] * 50.0  + kernels[50] * 30.072434619351  + kernels[51] * 6.51499589673  + kernels[52] * 34.064584367957  + kernels[53] * 50.0  + kernels[54] * 50.0  + kernels[55] * 50.0  + kernels[56] * 50.0  + kernels[57] * 50.0  + kernels[58] * 50.0  + kernels[59] * 50.0 );

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