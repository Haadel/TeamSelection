/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

import java.util.ArrayList;

/**
 *
 * @author ACER
 */
public class GA {
    public Problem problem;
    public Data data;
    public NSGA2 utils;
    public Population population;
    public GA(Problem problem, Data data){
        this.problem = problem;
        this.data = data;
        this.utils = new NSGA2(problem, data);
    }
     public ArrayList<Solution> Search(){
        this.population = utils.initial_population();
        
        utils.fast_nondominated_sort(population);
        for (ArrayList<Solution> front:population.fronts){
            utils.calculate_crowding_distance(front);
        }
        
        ArrayList<Solution> children = utils.create_children(population);
        Population return_population = null;
        for (int i=0;i<1000;i++){
            System.out.println(i);            
            population.extend(children);            
            utils.fast_nondominated_sort(population);
            Population new_population = new Population();
            int front_num = 0;
            while (new_population.population.size()+population.fronts.get(front_num).size()<1000){
                utils.calculate_crowding_distance(population.fronts.get(front_num));
                new_population.extend(population.fronts.get(front_num));
                front_num++;
//                System.out.println(population.fronts.get(front_num).size()+" "+ front_num);
            }
            utils.calculate_crowding_distance(population.fronts.get(front_num));
            population.fronts.get(front_num).sort((o1,o2)->{
                int flag = 0;
                if (o1.crowding_distance<o2.crowding_distance) flag = 1;
                if (o1.crowding_distance>o2.crowding_distance) flag = -1;
                return flag;
            });
            
            for (int j=0;j<500-new_population.population.size();j++){
                new_population.add(population.fronts.get(front_num).get(j));
            }
            return_population = population;
            population = new_population;
            utils.fast_nondominated_sort(population);
            for (ArrayList<Solution> front:population.fronts){
                utils.calculate_crowding_distance(front);
            }
        
            children = utils.create_children(population);
        }
        return return_population.fronts.get(0);
    }
}
