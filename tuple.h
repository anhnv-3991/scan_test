//#define SZ_PAGE 40960
//#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
//#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
//#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_X 256

#define MATCH_RATE 0.1

#define JT_SIZE 120000000
#define SELECTIVITY 1000000000

#define VAL_NUM 1

int right,left;

typedef struct _TUPLE {
  int key;
  int val;
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int lkey;
  int lval;
} RESULT;
/*
typedef struct _IDX {
  int val;
  int adr;
  struct _IDX *nxt;
} IDX;
*/
typedef struct _BUCKET {

    int val;
    int adr;

} BUCKET;
