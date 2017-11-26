/*
* Copyright (c) 2016 Jean-Noel Braun.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <bh/bh.h>
#include <bh/bh_timer.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"

int show_usage()
{
    fprintf(stderr, "Usage: pack-img <list_img_labels> <output_path> [-l label_width] [-t label_type]\n");
    fprintf(stderr, "\t Values for 'label_type': 'int' or 'float' or 'img' \n");
    return 0;
}

int is_option(char *argv)
{
    if (strncmp(argv, "-", 1) == 0)
        return 1;
    else
        return 0;
}

void bad_parameter(char *param)
{
    fprintf(stderr, "[ERROR] Bad parameter for option '%s'\n", param);
    show_usage();
}

void bad_option(char *opt)
{
    fprintf(stderr, "[ERROR] Unknown option '%s'\n", opt);
    show_usage();
}


int main(int argc, char **argv)
{
    int label_width = 1;
    bcnn_label_type type = LABEL_INT;
    int i = 3;

    if (argc < 2) {
        show_usage();
        return -1;
    }
    while (i < argc) {
        if (strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                if (is_option(argv[i + 1])) {
                    bad_parameter(argv[i]);
                    return -1;
                }
                label_width = atoi(argv[i + 1]);
            }
            else {
                bad_parameter(argv[i]);
                return -1;
            }
            i++;
        }
        else if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                if (is_option(argv[i + 1])) {
                    bad_parameter(argv[i]);
                    return -1;
                }
                if (!strcmp(argv[i + 1], "int")) type = LABEL_INT;
                else if (!strcmp(argv[i + 1], "float")) type = LABEL_FLOAT;
                else if (!strcmp(argv[i + 1], "img")) type = LABEL_IMG;
                else {
                    bad_parameter(argv[i]);
                    return -1;
                }
            }
            else {
                bad_parameter(argv[i]);
                return -1;
            }
            i++;
        }
        else {
            bad_option(argv[i]);
            return -1;
        }
        i++;
    }

    bcnn_pack_data(argv[1], label_width, type, argv[2]);

    return 0;
}


