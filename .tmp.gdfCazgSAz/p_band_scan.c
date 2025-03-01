#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "filter.h"
#include "signal.h"
#include "timing.h"


#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0

typedef struct {
    int tid; // thread id / name
    int band_start; // assign band range for each thread
    int band_end;
    int filter_order;
    double* band_power;
    signal* sig;
    double bandwidth;
    int num_threads;
    int num_bands;
} thread_data_t;

void usage() {
  printf("usage: band_scan text|bin|mmap signal_file Fs filter_order num_bands num_procs num_threads\n");
}

double avg_power(double* data, int num) {

  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }

  return ss / num;
}

double max_of(double* data, int num) {

  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {

  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {

  double dc = avg_of(data,num);

  printf("Removing DC component of %lf\n",dc);

  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}

#pragma GCC optimize("fast-math")
void* analyze_signal_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int tid = data->tid;
    int num_threads = data->num_threads;
    int num_bands = data->num_bands;
    int length = data->sig->num_samples;
    
    // calculate this thread's section
    int section_size = length / num_threads;
    int start = tid * section_size;
    int end = (tid == num_threads - 1) ? length : start + section_size;
    
    //printf("Thread %d processing samples %d to %d for all bands\n", tid, start, end-1);
    
    #pragma GCC ivdep
    for (int band = 0; band < num_bands; band++) {
        double filter_coeffs[data->filter_order + 1];
        
        generate_band_pass(data->sig->Fs,
                        band * data->bandwidth + 0.0001,
                        (band + 1) * data->bandwidth - 0.0001,
                        data->filter_order,
                        filter_coeffs);

        hamming_window(data->filter_order, filter_coeffs);

        // get power for this thread's section
        double pow_sum = 0;
        for (int i = start; i < end; i++) {
            double cur_sum = 0;

            if (data->filter_order >= 3) {
                // use vectors to speed up
                __m256d sum_vec = _mm256_setzero_pd();

                int j;
                #pragma GCC ivdep 
                for (j = data->filter_order; j >= 0; j -= 4) {
                    if ((i - j) >= 0 && (i - j - 3) >= 0) {
                        __m256d data_vec = _mm256_set_pd(data->sig->data[i - j], data->sig->data[i - j + 1], data->sig->data[i - j + 2], data->sig->data[i - j + 3]);
                        __m256d coef_vec = _mm256_set_pd(filter_coeffs[j], filter_coeffs[j - 1], filter_coeffs[j - 2], filter_coeffs[j - 3]);
                        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(data_vec, coef_vec));
                    }
                }

                // is there left over data?
                #pragma GCC ivdep
                for (; j >= 0; j-=1) {
                    bool is_valid = (i - j) >= 0 && (i - j) < length;
                    cur_sum += data->sig->data[i - j] * filter_coeffs[j] * is_valid;
                }

                // apparently need to use this storeu_pd thing. it works without it but apparently its not supposed to
                double temp[4];
                _mm256_storeu_pd(temp, sum_vec);
                cur_sum = temp[0] + temp[1] + temp[2] + temp[3];
                
            } else {
                // Original scalar code for small filter_order

                #pragma GCC ivdep
                for (int j = data->filter_order; j >= 0; j--) {
                    bool is_valid = (i - j) >= 0 && (i - j) < length;
                    cur_sum += data->sig->data[i - j] * filter_coeffs[j] * is_valid ;
                  }
            }
            
            pow_sum += cur_sum * cur_sum;
        }

        #pragma omp critical
        data->band_power[band] += pow_sum;
    }

   // printf("Thread %d finished task 1\n", tid);
    pthread_exit(NULL);
}



int analyze_signal(signal* sig, int filter_order, int num_bands, double* lb, double* ub, int num_procs, int num_threads) {

  // base stuff
  double Fc = (sig->Fs) / 2;
  double bandwidth = Fc / num_bands;
  remove_dc(sig->data,sig->num_samples);
  double signal_power = avg_power(sig->data,sig->num_samples);
  printf("signal average power:     %lf\n", signal_power);
  resources rstart;
  get_resources(&rstart,THIS_PROCESS);
  double start = get_seconds();
  unsigned long long tstart = get_cycle_count();
  double band_power[num_bands];

  #pragma GCC ivdep
  for (int i = 0; i < num_bands; i++) {
        band_power[i] = 0.0;
  }
  
  // how many bands per thread?
  int bands_per_thread = num_bands / num_threads;
  int bands_remainder = num_bands % num_threads;

  // allocate memory for the threads
  pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
  thread_data_t* thread_data = (thread_data_t*)malloc(num_threads * sizeof(thread_data_t));

  for (int tid = 0; tid < num_threads; tid++) {
    thread_data[tid].tid = tid;

    thread_data[tid].band_start = tid * bands_per_thread + (tid < bands_remainder ? tid : bands_remainder);
    thread_data[tid].band_end = thread_data[tid].band_start + bands_per_thread + (tid < bands_remainder ? 1 : 0); 
    thread_data[tid].filter_order = filter_order;
    thread_data[tid].band_power = band_power;
    thread_data[tid].sig = sig;
    thread_data[tid].bandwidth = bandwidth;
    thread_data[tid].num_threads = num_threads;
    thread_data[tid].num_bands = num_bands;
    
    // run the thread on its bands
    pthread_create(&(threads[tid]), NULL, analyze_signal_thread, (void*)&thread_data[tid]);
  }

  // sync threads
  for (int tid = 0; tid < num_threads; tid++) {
    pthread_join(threads[tid], NULL);
  }

  // normalize
  #pragma GCC ivdep
  for (int i = 0; i < num_bands; i++) {
    band_power[i] /= sig->num_samples;
  }


  unsigned long long tend = get_cycle_count();
  double end = get_seconds();

  resources rend;
  get_resources(&rend,THIS_PROCESS);

  resources rdiff;
  get_resources_diff(&rstart, &rend, &rdiff);

  // Pretty print results
  double max_band_power = max_of(band_power,num_bands);
  double avg_band_power = avg_of(band_power,num_bands);
  int wow = 0;
  *lb = -1;
  *ub = -1;

  for (int band = 0; band < num_bands; band++) {
    double band_low  = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;

    printf("%5d %20lf to %20lf Hz: %20lf ",
           band, band_low, band_high, band_power[band]);

    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
      printf("*");
    }

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

      // band of interest
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (*lb < 0) {
          *lb = band * bandwidth + 0.0001;
        }
        *ub = (band + 1) * bandwidth - 0.0001;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }

    printf("\n");
  }

  printf("Resource usages:\n\
         User time        %lf seconds\n\
         System time      %lf seconds\n\
         Page faults      %ld\n\
         Page swaps       %ld\n\
         Blocks of I/O    %ld\n\
         Signals caught   %ld\n\
         Context switches %ld\n",
                rdiff.usertime,
                rdiff.systime,
                rdiff.pagefaults,
                rdiff.pageswaps,
                rdiff.ioblocks,
                rdiff.sigs,
                rdiff.contextswitches);

  printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
         "Note that cycle count only makes sense if the thread stayed on one core\n",
         tend - tstart, cycles_to_seconds(tend - tstart), timing_overhead());
  printf("Analysis took %lf seconds by basic timing\n", end - start);

  return wow;
}

int main(int argc, char* argv[]) {

  if (argc != 8) {
    usage();
    return -1;
  }

  char sig_type    = toupper(argv[1][0]);
  char* sig_file   = argv[2];
  double Fs        = atof(argv[3]);
  int filter_order = atoi(argv[4]);
  int num_bands    = atoi(argv[5]);
  int num_procs    = atoi(argv[6]);
  int num_threads  = atoi(argv[7]);

  assert(Fs > 0.0);
  assert(filter_order > 0 && !(filter_order & 0x1));
  assert(num_bands > 0);

  printf("type:     %s\n\
        file:     %s\n\
        Fs:       %lf Hz\n\
        order:    %d\n\
        bands:    %d\n\
        procs:    %d\n\
        threads:  %d\n",
                sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
                sig_file,
                Fs,
                filter_order,
                num_bands,
                num_procs,
                num_threads);

  printf("Load or map file\n");

  signal* sig;
  switch (sig_type) {
    case 'T':
      sig = load_text_format_signal(sig_file);
      break;

    case 'B':
      sig = load_binary_format_signal(sig_file);
      break;

    case 'M':
      sig = map_binary_format_signal(sig_file);
      break;

    default:
      printf("Unknown signal type\n");
      return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;

  double start = 0;
  double end   = 0;
  if (analyze_signal(sig, filter_order, num_bands, &start, &end, num_procs, num_threads)) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", start, end, (end + start) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);

  return 0;
}

