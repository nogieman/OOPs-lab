#include "sim.h"
#include "utils.h"


/* n-dimensional-index incrementer
 * ii is a n-dimensional index
 * limit_shape is the shape of the tensor that ii is indexing
 *
 * for example,
 * consider, limit_shape = [3,4,2]
 * first,
 *  ii = [0,0,0]
 * calling increment_shape on it makes it,
 *  ii = [0,0,1]
 * then,
 *  ii = [0,1,0]
 *  ii = [0,1,1]
 *  ii = [0,2,0]
 *  ii = [0,2,1]
 *  ii = [0,3,0]
 *  ii = [0,3,1]
 *  ii = [1,0,0]
 * and so on till
 *  ii = [2,3,1]
 */
void increment_shape(std::vector<int> &ii,
                     const std::vector<int> &limit_shape) {
  assert(ii.size() == limit_shape.size());
  int current_index = ii.size() - 1;
  while (current_index >= 0) {
    ii.at(current_index)++;
    if (ii.at(current_index) >= limit_shape.at(current_index)) {
      ii.at(current_index) = 0;
      current_index--;
    } else {
      break;
    }
  }
  if (ii.at(0) >= limit_shape.at(0)) {
    log_fatal("Cannot increment past limit_shape\n");
  }
}

int calc_shift_val(float inverted) {
  int shift_val = 16;
  int calib_scale = inverted * (1<<shift_val);
  while (calib_scale < 10) {
    shift_val++;
    calib_scale = inverted * (1<<shift_val);
  }
  while (calib_scale > (1<<15)) {
    shift_val--;
    calib_scale = inverted * (1<<shift_val);
  }
  return shift_val;
}

