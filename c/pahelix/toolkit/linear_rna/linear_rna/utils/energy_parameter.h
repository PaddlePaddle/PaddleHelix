#ifndef LINEAR_RNA_UTILS_ENERGY_PARAMETER_H
#define LINEAR_RNA_UTILS_ENERGY_PARAMETER_H

#ifndef VIE_INF
#define VIE_INF 10000000 // to be the same as in vienna
#endif
#ifndef NBPAIRS
#define NBPAIRS 7
#endif

extern double g_lxc37;
extern int g_ml_intern37;
extern int g_ml_closing37;
extern int g_ml_base37;
extern int g_max_ninio;
extern int g_ninio37;
extern int g_terminal_au37;  // lhuang: outermost pair is AU or GU; also used in tetra_loop triloop

extern char g_triloops[241];
extern int g_triloop37[2];

extern char g_tetraloops[281];

extern int g_tetraloop37[16];

extern char g_hexaloops[361];
extern int g_hexaloop37[4];

extern int g_stack37[NBPAIRS+1][NBPAIRS+1];
extern int g_hairpin37[31];
extern int g_bulge37[31];
extern int g_internal_loop37[31];
extern int g_mismatch_i37[NBPAIRS+1][5][5];
extern int g_mismatch_h37[NBPAIRS+1][5][5];
extern int g_mismatch_m37[NBPAIRS+1][5][5];
extern int g_mismatch_1ni37[NBPAIRS+1][5][5];
extern int g_mismatch_23i37[NBPAIRS+1][5][5];
extern int g_mismatch_ext37[NBPAIRS+1][5][5];
extern int g_dangle5_37[NBPAIRS+1][5];
extern int g_dangle3_37[NBPAIRS+1][5];

#endif // LINEAR_RNA_UTILS_ENERGY_PARAMETER_H
