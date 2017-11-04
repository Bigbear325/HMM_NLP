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

    // transition matrix
    private Matrix mle_A;

    // emission matrix
    private Matrix mle_B;

    // prior of pos tags
    private Matrix mle_pi;

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
	//private Matrix gamma;
	
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
	private int max_iters = 30;
	
	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	private double[] accuracy = new double[max_iters];

	public float mu;

	private int predict_count;
	
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
        labeled_corpus = _labeled_corpus;
        unlabeled_corpus = _unlabeled_corpus;

        num_postags = 0;
        pos_tags = new Hashtable<>();
        inv_pos_tags = new Hashtable<>();

        num_words = 0;
        vocabulary = new Hashtable<>();

        max_sentence_length = 0;

        for(int i = 0; i < labeled_corpus.size(); ++i){
            Sentence sentence = labeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, sentence.length());
            for(int j = 0; j < sentence.length(); ++j){
                Word word = sentence.getWordAt(j);

                String lemme = word.getLemme();
                Integer lemme_id = vocabulary.get(lemme);
                if(lemme_id == null){
                    vocabulary.put(lemme, num_words);
                    ++num_words;
                }

                String pos_tag = word.getPosTag();
                Integer tag_id = pos_tags.get(pos_tag);
                if(tag_id == null){
                    pos_tags.put(pos_tag, num_postags);
                    inv_pos_tags.put(num_postags, pos_tag);
                    ++num_postags;
                }
            }
        }

        for(int i = 0; i < unlabeled_corpus.size(); ++i){
            Sentence sentence = unlabeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, sentence.length());
            for(int j = 0; j < sentence.length(); ++j){
                Word word = sentence.getWordAt(j);

                String lemme = word.getLemme();
                Integer lemme_id = vocabulary.get(lemme);
                if(lemme_id == null){
                    vocabulary.put(lemme, num_words);
                    ++num_words;
                }
            }
        }
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
        A = new Matrix(num_postags, num_postags + 1);
        B = new Matrix(num_postags, num_words);
        pi = new Matrix(1, num_postags);

        mle_A = new Matrix(num_postags, num_postags + 1);
        mle_B = new Matrix(num_postags, num_words);
        mle_pi = new Matrix(1, num_postags);
	}

	public void updateLambda(){
        for(int i = 0; i < num_postags; ++i){
            for(int j = 0; j < num_postags + 1; ++j) {
                A.set(i, j, mu * mle_A.get(i, j) + (1-mu) * A.get(i, j));
            }
        }

        for(int i = 0; i < num_postags; ++i){
            for(int j = 0; j < num_words; ++j) {
                B.set(i, j, mu * mle_B.get(i, j) + (1-mu) * B.get(i, j));
            }
        }

        for(int tag_id = 0; tag_id < num_postags; ++tag_id) {
            pi.set(0, tag_id, mu * mle_pi.get(0, tag_id) + (1-mu) * pi.get(0, tag_id));
        }
    }

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
        int[] count_tags = new int[num_postags];
        for(Sentence sentence : labeled_corpus){
            int pre_tag_id = -1;
            for(int i = 0; i < sentence.length(); ++i){
                Word word = sentence.getWordAt(i);
                int tag_id = pos_tags.get(word.getPosTag()).intValue();
                int lemme_id = vocabulary.get(word.getLemme()).intValue();
                count_tags[tag_id]++;
                if(i == 0){
                    mle_pi.set(0, tag_id, mle_pi.get(0, tag_id) + 1);
                    pi.set(0, tag_id, pi.get(0, tag_id) + 1);
                }
                else{
                    mle_A.set(pre_tag_id, tag_id, mle_A.get(pre_tag_id, tag_id) + 1);
                    A.set(pre_tag_id, tag_id, A.get(pre_tag_id, tag_id) + 1);
                }
                mle_B.set(tag_id, lemme_id, mle_B.get(tag_id, lemme_id) + 1);
                B.set(tag_id, lemme_id, B.get(tag_id, lemme_id) + 1);
                pre_tag_id = tag_id;
                if(i == sentence.length() - 1){
                    mle_A.set(tag_id, num_postags, mle_A.get(tag_id, num_postags) + 1);
                    A.set(tag_id, num_postags, A.get(tag_id, num_postags) + 1);
                }
            }
        }

        for(int i = 0; i < num_postags; ++i){
            for(int j = 0; j < num_words; ++j) {
                mle_B.set(i, j, mle_B.get(i,j) + smoothing_eps);
                B.set(i, j, B.get(i,j) + smoothing_eps);
            }
        }

        //normalize
        double scale = 0;
        for(int tag_id = 0; tag_id < num_postags; ++tag_id) {
            mle_pi.set(0, tag_id, mle_pi.get(0, tag_id) / labeled_corpus.size());
            pi.set(0, tag_id, pi.get(0, tag_id) / labeled_corpus.size());
        }

        for(int tag_id = 0; tag_id < num_postags; ++tag_id){
            scale = 0;
            for(int post_tag_id = 0; post_tag_id < num_postags + 1; ++post_tag_id)
                scale += mle_A.get(tag_id, post_tag_id);
            for(int post_tag_id = 0; post_tag_id < num_postags + 1; ++post_tag_id) {
                mle_A.set(tag_id, post_tag_id, mle_A.get(tag_id, post_tag_id) / scale);
                A.set(tag_id, post_tag_id, A.get(tag_id, post_tag_id) / scale);
            }

            scale = 0;
            for(int lemme_id = 0; lemme_id < num_words; ++lemme_id)
                scale += mle_B.get(tag_id, lemme_id);
            for(int lemme_id = 0; lemme_id < num_words; ++lemme_id) {
                mle_B.set(tag_id, lemme_id, mle_B.get(tag_id, lemme_id) / scale);
                B.set(tag_id, lemme_id, B.get(tag_id, lemme_id) / scale);
            }
        }
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
        //Initialize Matrices
        alpha = new Matrix(num_postags, max_sentence_length);
        beta = new Matrix(num_postags, max_sentence_length);
        scales = new Matrix(2, max_sentence_length);

        digamma = new Matrix(num_postags, num_postags + 1);

        //gamma = new Matrix(new double[1][num_postags]);
        gamma_0 = new Matrix(1, num_postags);
        gamma_w = new Matrix(num_postags, num_words);

        mle();

        for(int iter = 0; iter < max_iters; ++iter){
            final int temiter = iter;
            //E-step
            log_likelihood[iter] = 0;

            Thread[] threads = new Thread[unlabeled_corpus.size()];
            int c = 0;
            for(Sentence s : unlabeled_corpus){
                double PO = expection(s);
                log_likelihood[iter] += PO;

                /*threads[c++] = new Thread(new Runnable()
                {
                    public void run()
                    {
                        double PO = expection(s);
                        log_likelihood[temiter] += PO;
                    }
                });
                threads[c - 1].start();*/
            }
            /*try {
                for (int i = 0; i < c; ++i) threads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }*/

            //M-step
            maximization();

            //System.out.println("iteration " + iter);
        }
	}
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public double predict() {
        v = new Matrix(num_postags, max_sentence_length);
        back_pointer = new Matrix(num_postags, max_sentence_length);
        pred_seq = new Matrix(unlabeled_corpus.size(), max_sentence_length);
        int correct = 0;
        int all = 0;

        predict_count = Math.min(2012, unlabeled_corpus.size());

        for(int i = 0; i < predict_count; ++i){
            Sentence s = unlabeled_corpus.get(i);
            all += s.length();
            int index = (int)viterbi(s);

            int k = s.length() - 1;
            while(k >= 0){
                pred_seq.set(i, k, index);
                //if(index == pos_tags.get(s.getWordAt(k).getPosTag()).intValue())correct++;
                index = (int)back_pointer.get(index, k);
                --k;
            }

        }

        return (double)correct/all;
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
        FileWriter fw = new FileWriter(outFileName);
        BufferedWriter bw = new BufferedWriter(fw);
        int correct = 0;
        int Sum = 0;
        for(int i = 0; i < predict_count; ++i){
            Sentence s = unlabeled_corpus.get(i);
            for(int j = 0; j < s.length(); ++j){
                bw.write(s.getWordAt(j).getLemme() + " ");
                bw.write(inv_pos_tags.get((int)pred_seq.get(i, j)) + "\n");
                if(s.getWordAt(j).getPosTag().equals(inv_pos_tags.get((int)pred_seq.get(i, j))))correct++;
                Sum++;
            }
            bw.write("\n");
        }
        bw.close();
        fw.close();
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
        FileWriter fw = new FileWriter(outFileName);
        BufferedWriter bw = new BufferedWriter(fw);

        for(int i = 0; i < max_iters; ++i){
            //bw.write(log_likelihood[i] + " " + accuracy[i]+"\n");
            bw.write(log_likelihood[i] + "\n");
        }

        /*bw.write("\nA\n");
        for(int i = 0; i < num_postags; ++i){
            double sum = 0;
            for(int j = 0; j < num_postags; ++j){
                sum += A.get(i,j);
                bw.write(A.get(i,j) + "\t");
            }
            bw.write("\n--------------"+sum+"-----------------\n");
        }*/

        bw.close();
        fw.close();
	}
	
	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expection(Sentence s) {
        double PO = forward(s);
        double PO1 = backward(s);

        for(int t = 0; t < s.length(); ++t){

            int lemme_id = vocabulary.get(s.getWordAt(t).getLemme()).intValue();

            for(int tag_id = 0; tag_id < num_postags; ++tag_id){

                double alpha_t_i = alpha.get(tag_id, t);
                double beta_t_j = beta.get(tag_id, t);

                double new_gamma_w = alpha_t_i * beta_t_j;
                gamma_w.set(tag_id, lemme_id, gamma_w.get(tag_id, lemme_id) + new_gamma_w);
                //gamma.set(0, tag_id, gamma.get(0, tag_id) + new_gamma_w);
                if(t == 0)gamma_0.set(0, tag_id, gamma_0.get(0, tag_id) + new_gamma_w);
            }

            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(t < s.length() - 1) {
                    int post_lemme_id = vocabulary.get(s.getWordAt(t + 1).getLemme()).intValue();
                    for (int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id) {

                        double alpha_t_i = alpha.get(tag_id, t);
                        double beta_t1_j = beta.get(post_tag_id, t + 1);

                        double a_i_j = A.get(tag_id, post_tag_id);
                        double b_j_t1 = B.get(post_tag_id, post_lemme_id);

                        double inc_digamma = alpha_t_i * a_i_j * b_j_t1 * beta_t1_j;

                        digamma.set(tag_id, post_tag_id, digamma.get(tag_id, post_tag_id) + inc_digamma);
                    }
                }
                else{

                    double alpha_t_i = alpha.get(tag_id, t);
                    double beta_t1_j = 1;

                    double a_i_j = A.get(tag_id, num_postags);
                    double b_j_t1 = 1;

                    double inc_digamma = alpha_t_i * a_i_j * b_j_t1 * beta_t1_j;

                    digamma.set(tag_id, num_postags, digamma.get(tag_id, num_postags) + inc_digamma);
                }
            }
        }

        return PO;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
        double scale = 0;

        for(int i = 0; i < num_postags; ++i){

            scale = smoothing_eps * (num_postags + 1);
            for(int j = 0; j < num_postags + 1; ++j){
                scale += digamma.get(i, j);
            }
            for(int j = 0; j < num_postags + 1; ++j){
                A.set(i, j, (digamma.get(i, j) + smoothing_eps )/ scale);
                digamma.set(i, j, 0);
            }

            scale = smoothing_eps * num_words;

            for(int j = 0; j < num_words; ++j){
                scale += gamma_w.get(i, j);
            }
            for(int j = 0; j < num_words; ++j){
                B.set(i,j,(gamma_w.get(i, j) + smoothing_eps)/scale);
                gamma_w.set(i, j, 0);
            }

            //gamma.set(0, i, 0);
        }

        scale = 0;
        for(int i = 0; i < num_postags; ++i){
            scale += gamma_0.get(0, i);
        }
        for(int i = 0; i < num_postags; ++i){
            pi.set(0, i, gamma_0.get(0, i) / scale);
            gamma_0.set(0, i, 0);
        }

        updateLambda();
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
        for(int i = 0; i < s.length(); ++i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == 0){
                    alpha.set(tag_id, i, pi.get(0, tag_id) * B.get(tag_id, lemme_id));
                }
                else{
                    double ele = 0;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; ++pre_tag_id){
                        ele += alpha.get(pre_tag_id, i-1) * A.get(pre_tag_id, tag_id) * B.get(tag_id, lemme_id);
                    }
                    alpha.set(tag_id, i, ele);
                }
            }
            double scale = 0;
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                scale += alpha.get(tag_id, i);
            }
            if(scale == 0)System.out.println(s.getWordAt(i).getLemme());
            scales.set(0, i, 1 / scale);
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                alpha.set(tag_id, i, alpha.get(tag_id, i) * scales.get(0, i));
            }
        }

        double res = 0;
        for(int i = 0; i < s.length(); ++i)
            res += Math.log(1 / scales.get(0, i));
        return res;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
        for(int i = s.length() - 1; i >= 0; --i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            int post_lemme_id = 0;
            if(i < s.length() - 1)post_lemme_id = vocabulary.get(s.getWordAt(i + 1).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == s.length() - 1){
                    beta.set(tag_id, i, A.get(tag_id, num_postags));
                }
                else{
                    double ele = 0;
                    for(int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id){
                        ele += beta.get(post_tag_id, i+1) * A.get(tag_id, post_tag_id) * B.get(post_tag_id, post_lemme_id);
                    }
                    beta.set(tag_id, i, ele);
                }
            }
            double scale = 0;
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                scale += beta.get(tag_id, i);
            }
            scales.set(1, i, 1 / scale);
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                beta.set(tag_id, i, beta.get(tag_id, i) * scales.get(1, i));
            }
        }

        double res = 0;
        for(int i = 0; i < s.length(); ++i)
            res += Math.log(1 / scales.get(1, i));
        return res;
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s) {
        for(int i = 0; i < s.length(); ++i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == 0){
                    v.set(tag_id, i, Math.log(pi.get(0, tag_id)) + Math.log(B.get(tag_id, lemme_id)));
                }
                else{
                    double Max = Double.NEGATIVE_INFINITY;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; ++pre_tag_id){
                        double ele = v.get(pre_tag_id, i-1) + Math.log(A.get(pre_tag_id, tag_id)) + Math.log(B.get(tag_id, lemme_id));
                        if(ele > Max){
                            Max = ele;
                            back_pointer.set(tag_id, i, pre_tag_id);
                        }
                    }
                    v.set(tag_id, i, Max);
                }
            }
        }

        int res = 0;
        double Max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < num_postags; ++i) {
            if(v.get(i, s.length() - 1)  > Max){
                res = i;
                Max = v.get(i, s.length() - 1);
            }
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

        float mu = 0;
        if (args.length > 4) {
            mu = Float.parseFloat(args[4]);
            predictionFileName += "_" + args[4] + ".txt";
            trainingLogFileName += "_" + args[4] + ".txt";
        }

		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		model.mu = mu;

		model.prepareMatrices();
		model.em();
		model.predict();
		model.outputPredictions(predictionFileName);
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName);
		}
	}
}
