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
	
	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];

    // \mu: a value in [0,1] to balance estimations from MLE and EM
    // \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM totally.
    private double mu = 0.8;
    
    //private int predict_count; //new add
    
    // transition matrix
    private Matrix mle_A;
    
    // emission matrix
    private Matrix mle_B;
    
    // prior of pos tags
    private Matrix mle_pi;
    
    
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
        
        //Convert incoming corpus
        labeled_corpus = _labeled_corpus;
        unlabeled_corpus = _unlabeled_corpus;
        
        //Define three hash tables
        pos_tags = new Hashtable<>();
        inv_pos_tags = new Hashtable<>();
        vocabulary = new Hashtable<>();
        
        //three int
        num_postags = 0;
        num_words = 0;
        max_sentence_length = 0;
        
        //pre-process labeled_corpus
        for(int i = 0; i < labeled_corpus.size(); ++i){
            //parse the sentence and get the max length, each sentence use getSentence
            Sentence getSentence = labeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, getSentence.length());
            //
            for(int j = 0; j < getSentence.length(); ++j){
                Word getWord = getSentence.getWordAt(j);
                
                if(vocabulary.get(getWord.getLemme()) == null){
                    vocabulary.put(getWord.getLemme(), num_words);
                    num_words++;//counting
                }
            
                
                //mark: pos_tags = new Hashtable<>();inv_pos_tags = new Hashtable<>();
                if(pos_tags.get(getWord.getPosTag()) == null){
                    pos_tags.put(getWord.getPosTag(), num_postags);
                    inv_pos_tags.put(num_postags, getWord.getPosTag());
                    num_postags++;//counting
                }
            }
        }
        
        //pre-process unlabed_corpus
        for(int i = 0; i < unlabeled_corpus.size(); ++i){
            Sentence getSentence = unlabeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, getSentence.length());
            for(int j = 0; j < getSentence.length(); ++j){
                Word getWord = getSentence.getWordAt(j);
                
                
                if(vocabulary.get(getWord.getLemme()) == null){
                    vocabulary.put(getWord.getLemme(), num_words);
                    num_words++;//counting
                }
            }
        }
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
        A = new Matrix(new double[num_postags][num_postags + 1]);
        B = new Matrix(new double[num_postags][num_words]);
        pi = new Matrix(new double[1][num_postags]);
        
        mle_A = new Matrix(new double[num_postags][num_postags + 1]);
        mle_B = new Matrix(new double[num_postags][num_words]);
        mle_pi = new Matrix(new double[1][num_postags]);
        
        
        
//        mle_A = new Matrix(new double[num_postags][num_postags + 1]);
        
	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
    
    // Maximum likelihood estimation
	public void mle() {
        int[] count_tags =  new int[num_postags];
        for(Sentence s : labeled_corpus){
            int pre_tag_id = -1;
            for(int word = 0; word < s.length(); ++word){
                Word getWord = s.getWordAt(word);
                //pos_tags hashmap, vocabulary hashmap
                int cur_tag_id = pos_tags.get(getWord.getPosTag()).intValue();
                int cur_word_id = vocabulary.get(getWord.getLemme()).intValue();
                count_tags[cur_tag_id]++;
                if(word == 0) {
                    pi.set(0, cur_tag_id, pi.get(0, cur_tag_id)+1);//count initial probability  pi
                    mle_pi.set(0, cur_tag_id, pi.get(0, cur_tag_id)+1);//count initial probability  pi
                }
                else {
                    mle_A.set(pre_tag_id, cur_tag_id, A.get(pre_tag_id, cur_tag_id) + 1);//count transition prob, [pre+ cur]
                    A.set(pre_tag_id, cur_tag_id, A.get(pre_tag_id, cur_tag_id) + 1);//count transition prob, [pre+ cur]
                }
                mle_B.set(cur_tag_id, cur_word_id, B.get(cur_tag_id, cur_word_id)+ 1);//emission prob
                B.set(cur_tag_id, cur_word_id, B.get(cur_tag_id, cur_word_id)+ 1);//emission prob
                
                pre_tag_id = cur_tag_id;
                
                if(word == s.length() - 1){
                    A.set(cur_tag_id, num_postags, A.get(cur_tag_id, num_postags) + 1);
                }
            }
        }
    
        //smoothing_eps B
        for(int i = 0; i < num_postags; i++){
            for(int j = 0; j < num_words; j++){
//                if(B.get(i,j) == 0){
                mle_B.set(i,j, B.get(i,j) + smoothing_eps);
                    B.set(i,j, B.get(i,j) + smoothing_eps);
//                }
            }
        }
        
        //normalize
        double A_scale = 0, B_scale = 0;
        int pi_scale = labeled_corpus.size();
   
        for(int i = 0; i < num_postags; i++){
            pi.set(0, i, pi.get(0, i) / pi_scale);   //normalized inital prob
            mle_pi.set(0, i, pi.get(0, i) / pi_scale);   //normalized inital prob
        
        
            //calculate A transistion prob
            for(int j = 0; j <= num_postags; j++){
                A_scale += A.get(i, j); // i = prov, j = cur
            }
            for(int j = 0; j <= num_postags; j++){
                mle_A.set(i , j, A.get(i, j) / A_scale);
                A.set(i , j, A.get(i, j) / A_scale);
            }
        
            //calcueate B emmission prob
            for(int j = 0; j < num_words; j++){
                B_scale += B.get(i,j);    //from tag i to words
            }
            for(int j = 0; j < num_words; j++){
                mle_B.set(i, j, B.get(i, j) / B_scale);
                B.set(i, j, B.get(i, j) / B_scale);
            }
        }
    }



	/**
	 * Main EM algorithm. 
	 */
    //expectation maximization
	public void em() {

        
        //supervised learning of the labeled corpus
        mle();
        
        //semi-supervised learning of the unlabeded corpus
        alpha = new Matrix(new double[num_postags][max_sentence_length]);
        beta = new Matrix(new double[num_postags][max_sentence_length]);
        
        scales = new Matrix(new double[2][max_sentence_length]);
        digamma = new Matrix(new double[num_postags][num_postags + 1]);
        gamma = new Matrix(new double[1][num_postags]);
        gamma_0 = new Matrix(new double[1][num_postags]);
        gamma_w = new Matrix(new double[num_postags][num_words]);
        
        for(int iter = 0; iter < max_iters; ++iter){
            //E-step
            log_likelihood[iter] = 0;
            for(Sentence s : unlabeled_corpus){
                double p_O = expection(s);
                log_likelihood[iter] += p_O;
            }

            //M-step
//            if(iter == 0)maximization();
            maximization();
        }

    }
    
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
    private int predict_count;
	public void predict() {
        v = new Matrix(new double[num_postags][max_sentence_length]);//vertebi metrix
        back_pointer = new Matrix(new double[num_postags][max_sentence_length]);
        pred_seq = new Matrix(new double[unlabeled_corpus.size()][max_sentence_length]);
        
        
        predict_count = Math.min(2012, unlabeled_corpus.size());
        for(int i = 0; i < predict_count; i++){
            Sentence s = unlabeled_corpus.get(i);
            
            int index = (int)viterbi(s);
            
            int k = s.length() - 1;
            while(k >= 0){
                pred_seq.set(i, k, index);
                index = (int)back_pointer.get(index, k);
                k--;
            }
        }
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
        System.out.println((double)correct / Sum);
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
            bw.write(log_likelihood[i] + "\n");
        }
        
        bw.close();
        fw.close();
	}
	
	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
    private double expection(Sentence s) {
        double p_O = forward(s);
        backward(s);
        
        for(int t = 0; t < s.length(); ++t){
            
            int word_id = vocabulary.get(s.getWordAt(t).getLemme()).intValue();
            
            for(int tag_id = 0; tag_id < num_postags; tag_id++){
                
                double alphaT_i = alpha.get(tag_id, t);
                double betaT_j = beta.get(tag_id, t);
                
                double gamma2w = alphaT_i * betaT_j;
                gamma_w.set(tag_id, word_id, gamma_w.get(tag_id, word_id) + gamma2w);
                gamma.set(0, tag_id, gamma.get(0, tag_id) + gamma2w);
                if(t == 0)gamma_0.set(0, tag_id, gamma_0.get(0, tag_id) + gamma2w);
            }
            
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(t < s.length() - 1) {
                    int post_lemme_id = vocabulary.get(s.getWordAt(t + 1).getLemme()).intValue();
                    for (int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id) {
                        
                        double alphaT_i = alpha.get(tag_id, t);
                        double beta_t1_j = beta.get(post_tag_id, t + 1);
                        
                        double a_i_j = A.get(tag_id, post_tag_id);
                        double b_j_t1 = B.get(post_tag_id, post_lemme_id);
                        
                        double inc_digamma = alphaT_i * a_i_j * b_j_t1 * beta_t1_j;
                        
                        digamma.set(tag_id, post_tag_id, digamma.get(tag_id, post_tag_id) + inc_digamma);
                    }
                }
                else{
                    
                    double alphaT_i = alpha.get(tag_id, t);
                    double beta_t1_j = 1;
                    
                    double a_i_j = A.get(tag_id, num_postags);
                    double b_j_t1 = 1;
                    
                    double inc_digamma = alphaT_i * a_i_j * b_j_t1 * beta_t1_j;
                    
                    digamma.set(tag_id, num_postags, digamma.get(tag_id, num_postags) + inc_digamma);
                }
            }
        }
        
        return p_O;
    }

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
    private void maximization() {
        for(int i = 0; i < num_postags; i++){
            double scale = 0;
            for(int j = 0; j < num_postags + 1; ++j){
                scale += digamma.get(i, j);
            }
            for(int j = 0; j < num_postags + 1; ++j){
                A.set(i, j, digamma.get(i, j) / scale);
                digamma.set(i, j, 0);
            }
            
            for(int j = 0; j < num_words; ++j){
                B.set(i, j, gamma_w.get(i, j) / gamma.get(0, i));
                gamma_w.set(i, j, 0);
            }
            
            gamma.set(0, i, 0);
        }
        
        double scale = 0;
        for(int i = 0; i < num_postags; ++i){
            scale += gamma_0.get(0, i);
        }
        for(int i = 0; i < num_postags; ++i){
            pi.set(0, i, gamma_0.get(0, i) / scale);
            gamma_0.set(0, i, 0);
        }
        //updateLambda?
//        alpha = new Matrix(new double[num_postags][max_sentence_length]);
//        beta = new Matrix(new double[num_postags][max_sentence_length]);
//
//        scales = new Matrix(new double[2][max_sentence_length]);
//        digamma = new Matrix(new double[num_postags][num_postags + 1]);
//        gamma = new Matrix(new double[1][num_postags]);
//        gamma_0 = new Matrix(new double[1][num_postags]);
//        gamma_w = new Matrix(new double[num_postags][num_words]);
        
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
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */

    private double forward(Sentence s) {
        double index = 0;
        for(int word_order_in_s = 0; word_order_in_s < s.length(); ++word_order_in_s){
            int word_id = vocabulary.get(s.getWordAt(word_order_in_s).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(word_order_in_s == 0){
                    alpha.set(tag_id, word_order_in_s, pi.get(0, tag_id) * B.get(tag_id, word_id));
                }
                else{
                    double e_sum = 0;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; ++pre_tag_id){
                        e_sum += alpha.get(pre_tag_id, word_order_in_s-1) * A.get(pre_tag_id, tag_id) * B.get(tag_id, word_id);
                    }
                    alpha.set(tag_id, word_order_in_s, e_sum);
                }
            }
            double scalesum = 0;
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                scalesum += alpha.get(tag_id, word_order_in_s);
            }

            scales.set(0, word_order_in_s, scalesum);
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                alpha.set(tag_id, word_order_in_s, alpha.get(tag_id, word_order_in_s) / scales.get(0, word_order_in_s));
            }
        }
        

        for(int i = 0; i < s.length(); ++i)
            index += Math.log(1 / scales.get(0, i));
        return index;
    }


    /**
     * Backward algorithm for one sentence
     *
     * return: log P(O|\lambda)
     */
    private double backward(Sentence s) {
        double index = 0;

        for(int word_order_in_s = s.length() - 1; word_order_in_s >= 0; word_order_in_s--){
            int word_id = vocabulary.get(s.getWordAt(word_order_in_s).getLemme()).intValue();

            int pre_word_id = 0;
            if(word_order_in_s < s.length() - 1){
                pre_word_id = vocabulary.get(s.getWordAt(word_order_in_s + 1).getLemme()).intValue();
            }

            for(int tag_id = 0; tag_id < num_postags; tag_id++){
                if(word_order_in_s == s.length() -1){
                    //copy from A(af)
                    beta.set(tag_id, word_order_in_s, A.get(tag_id, num_postags));
                }
                else{
                    double ele_sum = 0;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; pre_tag_id++){
                        ele_sum += beta.get(pre_tag_id, word_order_in_s + 1) * A.get(tag_id, pre_tag_id) * B.get(pre_tag_id, pre_word_id);
                    }
                    beta.set(tag_id, word_order_in_s, ele_sum);
                }
            }

            double scale = 0;
            for(int tag_id = 0; tag_id < num_postags; tag_id++){
                scale += beta.get(tag_id, word_order_in_s);
            }
            scales.set(1, word_order_in_s, scale);
            for(int tag_id = 0; tag_id < num_postags; tag_id++){
                beta.set(tag_id, word_order_in_s, beta.get(tag_id,word_order_in_s) / scales.get(1,word_order_in_s));
            }
        }

        for(int i = 0; i < s.length(); i++){
            index += Math.log(scales.get(1,i));
        }
        return index;
    }

    
    
    
	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
    
    //v = max vi-1  * vij * bj(ot)
	private double viterbi(Sentence s) {
        for(int word_order_in_s = 0; word_order_in_s < s.length(); word_order_in_s++){
            int word_id = vocabulary.get(s.getWordAt(word_order_in_s).getLemme()).intValue();
            
            for(int tag_id = 0; tag_id < num_postags; tag_id++){ //cur_tag_id
                if(word_order_in_s == 0){
                    v.set(tag_id, word_order_in_s, Math.log(pi.get(0, tag_id)) + Math.log(B.get(tag_id, word_id))); // from: log pi+ log B (pi*b)
                }
                else{
                    double max_dp = Double.NEGATIVE_INFINITY;
                    //max dp : v = ai-1 * aij * B
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; pre_tag_id++){ //pre tag id
                        double dp = v.get(pre_tag_id, word_order_in_s - 1) + Math.log(A.get(pre_tag_id,tag_id)) + Math.log(B.get(tag_id, word_id));
                        if(dp > max_dp){
                            max_dp = dp;
                            back_pointer.set(tag_id, word_order_in_s, pre_tag_id);
                        }
                    }
                    v.set(tag_id, word_order_in_s , max_dp); //update
                }
            }
        }
        //choose the max v for index
        int index = 0;
        double Max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < num_postags; i++){
            if(v.get(i, s.length()-1) > Max){
                index = i;
                Max = Math.log(v.get(i, s.length()-1));
            }
        }
        
        return index;
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
            predictionFileName += "_" + args[4] + ".txt";
            trainingLogFileName += "_" + args[4] + ".txt";
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
        model.outputPredictions(predictionFileName);
        
        if (trainingLogFileName != null) {
            model.outputTrainingLog(trainingLogFileName);
        }
    }
}
