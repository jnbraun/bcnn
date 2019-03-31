/*
 * Copyright (c) 2016-present Jean-Noel Braun.
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

#ifndef BH_LOG_H
#define BH_LOG_H

#include <stdio.h> /* printf */
#include <stdlib.h>

#include <bh/bh_string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef bh_free
#define bh_free(buf)      \
    {                     \
        if (buf) {        \
            free(buf);    \
            (buf) = NULL; \
        }                 \
    }
#endif

typedef struct {
    char *name;
    char *val;
} bh_ini_parser_key;

typedef struct {
    int num_keys;
    char *name;
    bh_ini_parser_key *keys;
} bh_ini_parser_section;

typedef struct {
    int num_sections;
    bh_ini_parser_section *sections;
} bh_ini_parser;

static inline int bh_ini_parser_read_key(bh_ini_parser_section *section,
                                         const char *line) {
    if (section == NULL) {
        fprintf(stderr, "[ERROR] No valid section for key %s\n", line);
    }
    char **tok = NULL;
    int num_toks = bh_strsplit((char *)line, '=', &tok);
    if (num_toks != 2) {
        fprintf(stderr, "[ERROR] Invalid key section %s\n", line);
        return -1;
    }
    bh_ini_parser_key key = {0};
    bh_strfill(&key.name, tok[0]);
    bh_strfill(&key.val, tok[1]);
    for (int i = 0; i < num_toks; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);
    section->num_keys++;
    bh_ini_parser_key *p_keys = (bh_ini_parser_key *)realloc(
        section->keys, section->num_keys * sizeof(bh_ini_parser_key));
    if (p_keys == NULL) {
        fprintf(stderr, "[ERROR] Failed allocation\n");
        return -1;
    }
    section->keys = p_keys;
    section->keys[section->num_keys - 1] = key;
    return 0;
}

static inline int bh_ini_parser_read_section(bh_ini_parser *config,
                                             const char *line) {
    bh_ini_parser_section section = {0};
    config->num_sections++;
    bh_ini_parser_section *p_sections = (bh_ini_parser_section *)realloc(
        config->sections, config->num_sections * sizeof(bh_ini_parser_section));
    if (p_sections == NULL) {
        fprintf(stderr, "[ERROR] Failed allocation\n");
        return -1;
    }
    bh_strfill(&section.name, line);
    config->sections = p_sections;
    config->sections[config->num_sections - 1] = section;
    return 0;
}

static inline void bh_ini_parser_destroy(bh_ini_parser *config) {
    for (int i = 0; i < config->num_sections; ++i) {
        for (int j = 0; j < config->sections[i].num_keys; ++j) {
            bh_free(config->sections[i].keys[j].name);
            bh_free(config->sections[i].keys[j].val);
        }
        bh_free(config->sections[i].keys);
        bh_free(config->sections[i].name);
    }
    bh_free(config->sections);
    bh_free(config);
}

static inline bh_ini_parser *bh_ini_parser_create(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "[ERROR] Could not open file: %s\n", filename);
        return NULL;
    }
    char *line = NULL;
    bh_ini_parser *config = (bh_ini_parser *)calloc(1, sizeof(bh_ini_parser));
    bh_ini_parser_section *section = NULL;
    while ((line = bh_fgetline(file)) != NULL) {
        bh_strstrip(line);
        switch (line[0]) {
            case '[': {
                if (bh_ini_parser_read_section(config, line) != 0) {
                    fprintf(stderr, "[ERROR] Failed to parse config file %s\n",
                            filename);
                    bh_free(line);
                    bh_ini_parser_destroy(config);
                    return NULL;
                }
                section = &config->sections[config->num_sections - 1];
                break;
            }
            case '!':
            case '\0':
            case '#':
            case ';':
                break;
            default: {
                if (bh_ini_parser_read_key(section, line) != 0) {
                    fprintf(stderr, "[ERROR] Failed to parse config file %s\n",
                            filename);
                    bh_free(line);
                    bh_ini_parser_destroy(config);
                    return NULL;
                }
                break;
            }
        }
        bh_free(line);
    }
    fclose(file);
    return config;
}

#ifdef __cplusplus
}
#endif

#endif  // BH_LOG_H