#include <bip/bip.h>

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

static int rand_between(int min, int max) {
    if (min > max) {
        return 0.f;
    }
    return (int)(((float)rand() / RAND_MAX * (max - min)) + min + 0.5);
}

static float frand_between(float min, float max) {
    if (min > max) {
        return 0.f;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min + 0.5;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in> <out> <brightness>\n", argv[0]);
        return -1;
    }
    unsigned char *img = NULL;
    int w, h, c;
    if (bip_load_image(argv[1], &img, &w, &h, &c) != BIP_SUCCESS) {
        fprintf(stderr, "Could not open image %s\n", argv[1]);
        bh_free(img);
        return -1;
    }
    unsigned char *out =
        (unsigned char *)calloc(w * h * c, sizeof(unsigned char));
    memcpy(out, img, w * h * c);

    for (int i = 0; i < rand_between(1, 3); ++i) {
        int mu_x = rand_between(w / 2 - w / 3, w / 2 + w / 3);
        int mu_y = rand_between(h / 2 - h / 3, h / 2 + h / 3);
        int sz_x = rand_between(0, 2);
        int sz_y = rand_between(0, 2);
        float sigma_x = frand_between(0.2f, 1.5f);
        float sigma_y = frand_between(0.2f, 1.5f);
        float inv_sigma2_x = 1 / (sigma_x * sigma_x);
        float inv_sigma2_y = 1 / (sigma_y * sigma_y);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float val = 0.f;
                /*for (int k = -sz_y / 2; k < sz_y / 2 + 1; ++k) {
                    for (int l = -sz_x / 2; l < sz_x / 2 + 1; ++l) {
                        val = bh_max(
                            val, exp(-0.5f * (inv_sigma2_x * (x - (mu_x + l)) *
                                                  (x - (mu_x + l)) +
                                              inv_sigma2_y * (y - (mu_y + k)) *
                                                  (y - (mu_y + k)))));
                    }
                }*/
                val = exp(-0.5f * (inv_sigma2_x * (x - (mu_x)) * (x - (mu_x)) +
                                   inv_sigma2_y * (y - (mu_y)) * (y - (mu_y))));
                if (val > 0.0f) {
                    out[y * w + x] = (unsigned char)bh_clamp(
                        255.0f * val + out[y * w + x], 0, 255);
                }
            }
        }
    }

    bip_write_image(argv[2], out, w, h, c, w * c);
    bh_free(img);
    bh_free(out);
    return 0;
}
