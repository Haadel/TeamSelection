/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

/**
 *
 * @author ACER
 */
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
public class Population {
    public ArrayList<Solution> population ;
    public ArrayList<ArrayList<Solution>> fronts;

    public Population() {
        this.population = new ArrayList<Solution>();
        this.fronts = new ArrayList<ArrayList<Solution>>();
    }

    public void add(Solution new_individual){
        this.population.add(new_individual);
    }
    
    public void extend(ArrayList<Solution> new_individuals){
        this.population.addAll(new_individuals);
    }
    
    public void sortChromosomesByFitness()
    {
        Collections.sort(this.population,(chromosome1 , chromosome2)->{
            int flag=0;
            if(chromosome1.fitness < chromosome2.fitness) flag = -1;
            if(chromosome1.fitness > chromosome2.fitness) flag = 1;
            return flag;
        });  
    }
}
