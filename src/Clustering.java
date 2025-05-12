import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;

public class Clustering {

    public static void main(String args[]) throws Exception {

        //load dataset		
        DataSource source = new DataSource("data/breast-cancer-final.arff");
        
        //get instances object 
        Instances dataset = source.getDataSet();
        
        //set class index to first attribute for dataset
        dataset.setClassIndex(0);
        
        //generate data for clusterer (w/o class)
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (dataset.classIndex() + 1));
        filter.setInputFormat(dataset);
        Instances clusterData = Filter.useFilter(dataset, filter); //This is the new dataset without the class attribute.

        //create clusterer
        SimpleKMeans clusterer = new SimpleKMeans();
        
        //set clusterer options
        String[] options = new String[5];
        //Options Array
        options[0] = "-N";      //sets number of cluster to 2
        options[1] = "2"; 
        options[2] = "-init";   //sets Initialization to k-means++
        options[3] = "1";      
        options[4] = "-V";      //Display std. deviations for centroids
        
        //apply options
        clusterer.setOptions(options);
        
        //set distance function
        //model.setDistanceFunction(new weka.core.ManhattanDistance());
        
        // build the clusterer
        clusterer.buildClusterer(clusterData);
        
        //print clustering results
        //System.out.println(clusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(dataset);

        // print evaluation results
        System.out.println("==================================================================");
        System.out.println(eval.clusterResultsToString());
        System.out.println("==================================================================");
    }
}
