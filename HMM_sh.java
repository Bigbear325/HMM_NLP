import java.util.Set;
import java.util.Hashtable;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import Jama.Matrix;

class HMM {
	/* Section for variables regarding the data */
	
	//
	private ArrayList<Sentence> labeled_corpus;
	
	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;
	
	// mapping POS tags in String to their indices
	Hashtable<String, Integer> pos_tags;
	
	// inverse of pos_tags: mapping POS tag indices to their String format
	Hashtable<Integer, String> inv_pos_tags;
	
	// vocabulary size
	int num_words;

	Hashtable<String, Integer> vocabulary;

	private int max_sentence_length;
	
	/* Section for variables in HMM */
	
	// transition matrix
	private Matrix A;

	// emission matrix
	private Matrix B;

	// prior of pos tags
	private Matrix pi;

	// store the scaled alpha and beta
	private Matrix alpha;
	
	private Matrix beta;

	// scales to prevent alpha and beta from underflowing
	private Matrix scales;

	// logged v for Viterbi
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;
	
	// \xi_t(i): expected frequency of pos tag i at position t. Use as an accumulator.
	private Matrix gamma;
	
	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at position t.  Use as an accumulator.
	private Matrix digamma;
	
	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private Matrix gamma_w;

	// \xi_0(i): expected frequency of pos tag i at position 0.
	private Matrix gamma_0;
	
	/* Section of parameters for running the algorithms */

	// smoothing epsilon for the B matrix (since there are likely to be unseen words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.1;

	// number of iterations of EM
	private int max_iters = 10;

	// \mu: a value in [0,1] to balance estimations from MLE and EM
	// \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM totally.
	private double mu = 0.8;
	
	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
        labeled_corpus = _labeled_corpus;
        unlabeled_corpus = _unlabeled_corpus;
        vocabulary = new Hashtable<>();
        pos_tags = new Hashtable<>();
        inv_pos_tags = new Hashtable<>();
	}

	/**
	 * Set the semi-supervised parameter \mu
	 */
	public void setMu(double _mu) {
		if (_mu < 0) {
			this.mu = 0.0;
		} else if (_mu > 1) {
			this.mu = 1.0;
		}
		this.mu = _mu;
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
        num_postags = 0;
        num_words = 0;
        max_sentence_length = Integer.MIN_VALUE;
        for (Sentence s: labeled_corpus) {
            max_sentence_length = Integer.max(max_sentence_length, s.length());
            for (Word word: s) {
                if (!vocabulary.containsKey(word.getLemme()))
                    vocabulary.put(word.getLemme(), num_words++);

                if (!pos_tags.containsKey(word.getPosTag())) {
                    pos_tags.put(word.getPosTag(), num_postags);
                    inv_pos_tags.put(num_postags, word.getPosTag());
                    ++num_postags;
                }
            }
        }

        for (Sentence s: unlabeled_corpus) {
            max_sentence_length = Integer.max(max_sentence_length, s.length());
            for (Word word: s) {
                if (!vocabulary.containsKey(word.getLemme()))
                    vocabulary.put(word.getLemme(), num_words++);

                if(word.getPosTag() != null) {
                    if (!pos_tags.containsKey(word.getPosTag())) {
                        pos_tags.put(word.getPosTag(), num_postags);
                        inv_pos_tags.put(num_postags, word.getPosTag());
                        ++num_postags;
                    }
                }
            }
        }

        A = new Matrix(num_postags, num_postags+1);
        B = new Matrix(num_postags, num_words);
        pi = new Matrix(num_postags, 1);

        alpha = new Matrix(num_postags, max_sentence_length);
        beta = new Matrix(num_postags, max_sentence_length);
        scales = new Matrix(2, max_sentence_length);

        gamma = new Matrix(num_postags, 1);
        digamma = new Matrix(num_postags, num_postags+1);
        gamma_w = new Matrix(num_postags, num_words);
        gamma_0 = new Matrix(num_postags, 1);

        v = new Matrix(num_postags, max_sentence_length);
        back_pointer = new Matrix(num_postags, max_sentence_length);
        pred_seq = new Matrix(unlabeled_corpus.size(), max_sentence_length);
	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
        // initialize PI
        int num_sentences = labeled_corpus.size();
        double[] count_postags = new double[num_postags];
        for (Sentence s : labeled_corpus) {
            for (int i = 0; i < s.length(); ++i) {
                Word word = s.getWordAt(i);
                int word_idx = vocabulary.get(word.getLemme());
                int postag_idx = pos_tags.get(word.getPosTag());
                ++count_postags[postag_idx];
                B.set(postag_idx, word_idx, B.get(postag_idx, word_idx)+1);
                if (i == 0) {
                    pi.set(postag_idx, 0, pi.get(postag_idx, 0) + 1);
                }
                else if (i == s.length() - 1) {
                    A.set(postag_idx, num_postags, A.get(postag_idx, num_postags) + 1);
                }
                else {
                    Word word_i = s.getWordAt(i), word_j = s.getWordAt(i + 1);
                    int idx_i = pos_tags.get(word_i.getPosTag()), idx_j = pos_tags.get(word_j.getPosTag());
                    A.set(idx_i, idx_j, A.get(idx_i, idx_j) + 1);
                }
            }
        }

        for (int i = 0; i < num_postags; ++i) {
            pi.set(i, 0, pi.get(i, 0)/num_sentences);

            double sumB = 0.0;
            for (int j = 0; j < num_words; ++j) {
                if (B.get(i, j) == 0)
                    B.set(i, j, smoothing_eps);
                sumB += B.get(i, j);
            }
            for (int j = 0; j < num_words; ++j)
                B.set(i, j, B.get(i, j) / sumB);

            double sumA = 0.0;
            for (int j = 0; j < num_postags + 1; ++j) {
                if (A.get(i, j) == 0)
                    A.set(i, j, smoothing_eps);
                sumA += A.get(i, j);
            }

            for (int j = 0; j < num_postags + 1; ++j)
                A.set(i, j, A.get(i, j) / sumA);
        }
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
        mle();

        for(int iter = 0; iter < max_iters; ++iter){
            //System.out.println(iter + " iteration");
            double sum = 0;
            //Expectation
            for(Sentence s : unlabeled_corpus) {
                double tmp = expectation(s);
                //System.out.println(tmp);
                sum += tmp;
            }
            log_likelihood[iter] = sum;
            //Maximization
            maximization();
        }

	}
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
        /*for (int i = 0; i < unlabeled_corpus.size(); ++i)
            viterbi(unlabeled_corpus.get(i), i);*/
        for (int i = 0; i < 2012; ++i)
            viterbi(unlabeled_corpus.get(i), i);
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
        FileWriter fw = new FileWriter(outFileName);
        BufferedWriter bw = new BufferedWriter(fw);
        double count = 0, correct_count = 0;
        for (int i = 0; i < 2012; ++i) {
            Sentence s = unlabeled_corpus.get(i);
            for (int t = 0; t < s.length(); ++t) {
                ++count;
                Word word = s.getWordAt(t);
                int postag_id = (int)pred_seq.get(i, t);
                String postag = inv_pos_tags.get(postag_id);
                bw.write(word.getLemme() + " " + postag + "\n");
                //if (postag.equals(word.getPosTag())) ++correct_count;
            }
            bw.write("\n");
        }
        bw.close();
        fw.close();
       //System.out.println(correct_count/count);
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
        FileWriter fw = new FileWriter(outFileName);
        BufferedWriter bw = new BufferedWriter(fw);

        for (int i = 0; i < max_iters; ++i) {
            bw.write(log_likelihood[i] + "\n");
        }

        bw.close();
        fw.close();
	}
	
	/**
	 * Expectation step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expectation(Sentence s) {
        double PO = forward(s);
        backward(s);
        int T = s.length();
        for (int t = 0; t < s.length(); ++t) {
            Word word = s.getWordAt(t);
            int word_idx = vocabulary.get(word.getLemme());

            for (int j = 0; j < num_postags; ++j) {
                double next_gamma = alpha.get(j, t) * beta.get(j, t);
                if (t == 0)
                    gamma_0.set(j, 0, gamma_0.get(j, 0) + next_gamma);
                gamma.set(j, 0, gamma.get(j, 0) + next_gamma);
                gamma_w.set(j, word_idx, gamma_w.get(j, word_idx) + next_gamma);
            }

            for (int i = 0; i < num_postags; ++i) {
                if (t < T - 1) {
                    Word next_word = s.getWordAt(t+1);
                    int next_word_idx = vocabulary.get(next_word.getLemme());
                    for (int j = 0; j < num_postags; ++j) {
                        double next_gamma = alpha.get(i, t) * A.get(i, j) * B.get(j, next_word_idx) * beta.get(j, t+1);
                        digamma.set(i, j, digamma.get(i, j) + next_gamma);
                    }
                }
                else {
                    double next_gamma = alpha.get(i, t) * A.get(i, num_postags);
                    digamma.set(i, num_postags, digamma.get(i, num_postags) + next_gamma);
                }
            }

        }
        return PO;
	}

    private void resetMatrices() {
        alpha = new Matrix(num_postags, max_sentence_length);
        beta = new Matrix(num_postags, max_sentence_length);
        scales = new Matrix(2, max_sentence_length);

        gamma = new Matrix(num_postags, 1);
        digamma = new Matrix(num_postags, num_postags+1);
        gamma_w = new Matrix(num_postags, num_words);
        gamma_0 = new Matrix(num_postags, 1);
    }

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
	    // normalize gamma matrices
        for (int i = 0; i < num_postags; ++i) {
            double sum = 0;
            for (int j = 0; j < num_postags+1; ++j)
                sum += digamma.get(i, j);

            for (int j = 0; j < num_postags+1; ++j) {
                if (sum == 0)
                    A.set(i, j, mu * A.get(i, j) + (1 - mu) * (1 / (num_postags + 1)));
                else
                    A.set(i, j, mu * A.get(i, j) + (1 - mu) * digamma.get(i, j) / sum);
            }


            for (int j = 0; j < num_words; ++j)
                if (gamma.get(i, 0) == 0)
                    B.set(i, j, mu*B.get(i, j) + (1-mu)*(1 / num_words));
                else
                    B.set(i, j, mu*B.get(i, j) + (1-mu)*(gamma_w.get(i, j) / gamma.get(i, 0)));

            if (gamma.get(i, 0) == 0)
                pi.set(i, 0, mu*pi.get(i, 0) + (1-mu)*(1/num_postags));
            else
                pi.set(i, 0, mu*pi.get(i, 0) + (1-mu)*(gamma_0.get(i, 0) / gamma.get(i, 0)));
        }

        resetMatrices();
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
        for (int k = 0; k < s.length(); ++k) {
            Word word = s.getWordAt(k);
            int word_idx = vocabulary.get(word.getLemme());
            for (int j = 0; j < num_postags; ++j) {
                if (k == 0) {
                    alpha.set(j, k, pi.get(j, 0) * B.get(j, word_idx)); // initialization
                }
                else {
                    double sum = 0.0;
                    for (int i = 0; i < num_postags; ++i)
                        sum += alpha.get(i, k - 1) * A.get(i, j) * B.get(j, word_idx);
                    alpha.set(j, k, sum);

                }
            }

            double ct = 0.0;
            for (int j = 0; j < num_postags; ++j)
                ct += alpha.get(j, k);
            scales.set(0, k, 1/ct);
            for (int j = 0; j < num_postags; ++j)
                alpha.set(j, k, alpha.get(j, k) / ct); // normalization

        }
        double prob = 0.0;
        for (int t = 0; t < s.length(); ++t)
            prob += Math.log(1/scales.get(0, t));
        return  prob;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
        for (int k = s.length()-1; k >= 0; --k) {
            for (int i = 0; i < num_postags; ++i) {
                if (k == s.length() - 1)
                    beta.set(i, k, A.get(i, num_postags)); // initialization
                else {
                    int word_idx = vocabulary.get(s.getWordAt(k+1).getLemme());
                    double sum = 0.0;
                    for (int j = 0; j < num_postags; ++j)
                        sum += beta.get(j, k + 1) * A.get(i, j) * B.get(j, word_idx);
                    beta.set(i, k, sum);

                }
            }

            double ct = 0.0;
            for (int i = 0; i < num_postags; ++i)
                ct += beta.get(i, k);
            scales.set(1, k, 1/ct);
            for (int i = 0; i < num_postags; ++i)
                beta.set(i, k, beta.get(i, k) / ct); // normalization
        }
        double prob = 0.0;
        /*
        int word0_idx = vocabulary.get(s.getWordAt(0).getLemme());
        for (int j = 0; j < num_postags; ++j) {
            prob += beta.get(j, 0) * pi.get(j, 0) * B.get(j, word0_idx);
        }*/
        for (int t = 0; t < s.length(); ++t)
            prob += Math.log(1/scales.get(1, t));
        return  prob;
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s, int s_id) {
        int T = s.length();
        int word0_idx = vocabulary.get(s.getWordAt(0).getLemme());

        for (int pos_i = 0; pos_i < num_postags; ++pos_i) {
            v.set(pos_i, 0, Math.log(pi.get(pos_i, 0)) + Math.log(B.get(pos_i, word0_idx)));
            back_pointer.set(pos_i, 0, 0);
        }

        for (int t = 1; t < T; ++t) {
            int word_idx = vocabulary.get(s.getWordAt(t).getLemme());
            for (int pos_j = 0; pos_j < num_postags; ++pos_j) {
                double max_prob = Double.NEGATIVE_INFINITY;
                for (int pos_i = 0; pos_i < num_postags; ++pos_i) {
                    double prob = v.get(pos_i, t-1) + Math.log(A.get(pos_i, pos_j)) + Math.log(B.get(pos_j, word_idx));
                    if (prob > max_prob) {
                        max_prob = prob;
                        back_pointer.set(pos_j, t, pos_i);
                    }
                }
                v.set(pos_j, t, max_prob);
            }
        }

        double res = Double.NEGATIVE_INFINITY;
        int last_step = 0;
        for (int pos_i = 0; pos_i < num_postags; ++pos_i) {
            //double prob = v.get(pos_i, T-1) + Math.log(A.get(pos_i, num_postags));
            double prob = v.get(pos_i, T-1);
            if (prob > res) {
                res = prob;
                last_step = pos_i;
            }
        }

        int prev_tag = last_step;
        pred_seq.set(s_id, T-1, prev_tag);
        for (int t = T - 2; t >= 0; --t) {
            int next = (int)back_pointer.get(prev_tag, t+1);
            pred_seq.set(s_id, t, next);
            prev_tag = next;
        }

        return res;
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}
		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
		
		String trainingLogFileName = null;

		
		if (args.length > 3) {
			trainingLogFileName = args[3];
		}
		
		double mu = 0.8;
		
		if (args.length > 4) {
			mu = Double.parseDouble(args[4]);
		}
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		
		model.setMu(mu);
		
		model.prepareMatrices();
		
		model.em();
		model.predict();
		model.outputPredictions("results/p2/prediction" + "_" + String.format("%.1f", mu) + ".txt");
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
	}
}
