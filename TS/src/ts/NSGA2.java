/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Acer
 */
public class NSGA2 {
    public Problem problem;
    public Data data;
    
    public NSGA2(Problem problem, Data data){
        this.problem = problem;
        this.data = data;
    }
    
    public Population initial_population(){
        Population population = new Population();
        for (int i=0;i<1000;i++){
            Solution individual = problem.generate_solution();
            problem.cal_objectives(individual);
            population.add(individual);
        }
        return population;
    }
    
    public void fast_nondominated_sort(Population population){
        population.fronts = new ArrayList<ArrayList<Solution>>();
        ArrayList<Solution> lst = new ArrayList<Solution>();
        for (Solution individual:population.population){
            individual.domination_count = 0;
            individual.dominated_solution = new ArrayList<Solution>();
            
            for (Solution other_individual:population.population){
                if (individual.dominates(other_individual)){
                    individual.dominated_solution.add(other_individual);
                }else{
                    if (other_individual.dominates(individual)){
                        individual.domination_count++;
                    }
                }
            }
            
            
            
            if (individual.domination_count==0){
                individual.rank = 0;
                lst.add(individual);   
                
            }
            
            
        }
        population.fronts.add(0,lst);
//        System.out.println(population.fronts.get(0).size()+" aa");
//        for (Solution i:population.population){
//            System.out.println(i.domination_count+" "+i.dominated_solution.size());
//        }
        int i = 0;
        while (population.fronts.get(i).size()>0){
            lst = new ArrayList<Solution>();
            for (Solution individual:population.fronts.get(i)){
                for (Solution other_individual:individual.dominated_solution){
                    
                    other_individual.domination_count--;
                    
                    if (other_individual.domination_count==0){
                        other_individual.rank = i+1;
                        lst.add(other_individual);
                    }
                }
            i++;
            population.fronts.add(i,lst);
            }
        }  
    }
    
    public void calculate_crowding_distance(ArrayList<Solution> front){
        if (front.size()>0){
            int solution_num = front.size();
            
            for (Solution individual:front){
                individual.crowding_distance = 0;
            }
            
            for (int i=0;i<front.get(0).objectives.length;i++){
                final int idx = i;
                front.sort((o1,o2)->{
                    int flag = 0;
                    if (o1.objectives[idx]<o2.objectives[idx]) flag = -1;
                    if (o1.objectives[idx]>o2.objectives[idx]) flag = 1;
                    return flag;
                });
                
                front.get(0).crowding_distance = Math.pow(10, 9);
                front.get(solution_num-1).crowding_distance = Math.pow(10, 9);
                
                double max = -1;
                double min = -1;
                for (int j=0;j<front.size();j++){
                    double curr = front.get(j).objectives[i];
                    if ((max==-1) || (curr>max)){
                        max = curr;
                    }
                    
                    if ((min==-1) || (curr<min)){
                        min = curr;
                    } 
                }
                
                double scale = max-min;
                if (scale==0) scale =1;
                for (int j = 1; j<solution_num-1;j++){
                    front.get(j).crowding_distance+=(front.get(j+1).objectives[idx]-front.get(j-1).objectives[idx])/scale;
                }
                   
            }
        }
    }
    
    public int crowding_operator(Solution individual, Solution other_individual){
        if ((individual.rank<other_individual.rank) || ((individual.rank== other_individual.rank) && (individual.crowding_distance>other_individual.crowding_distance))){
            return 1;
        }else{
            return 0;
        }
    }
    
    public ArrayList<Solution> create_children(Population population){
        ArrayList<Solution> children = new ArrayList<Solution>();
                ArrayList<Solution> children1 = (ArrayList<Solution>) population.population.clone();
                Collections.sort(children1, (o1, o2) -> {
                    return -Double.compare(o1.fitness, o2.fitness); //To change body of generated lambdas, choose Tools | Templates.
                });
        for(int i =0;i<100;i++){
            children.add(children1.get(i));
        }        
        for(int i =0;i<900;i++){
            Random ran = new Random();
            int x = ran.nextInt(children1.size());
            int y = ran.nextInt(children1.size());
            while(y==x) y = ran.nextInt(children1.size());
            children.add(_crossover(children1.get(x), children1.get(y)));
        }
       Collections.sort(children, (o1, o2) -> {
                    return -Double.compare(o1.fitness, o2.fitness); //To change body of generated lambdas, choose Tools | Templates.
                });
        for(int i=0;i<50;i++){
            Random ran = new Random();
            int x = ran.nextInt(900)+50;
            children.set(x, problem.generate_solution());
        }
        return children;
    }
    
    public Solution _crossover(Solution parent1, Solution parent2){
        Solution child = new Solution(data);
        
        ArrayList<Integer> chosen =new ArrayList<>();
        ArrayList<Integer> candidate =new ArrayList<>();
        for(int g=0;g<data.numberOfGroup;g++){
            for(int i =0;i<data.groupNeed.get(g);i++){
                if(!chosen.contains(parent1.candi.get(g).get(i))){
                    candidate.add(parent1.candi.get(g).get(i));
                    chosen.add(parent1.candi.get(g).get(i));
                }
                if(!chosen.contains(parent2.candi.get(g).get(i))){
                    candidate.add(parent2.candi.get(g).get(i));
                    chosen.add(parent2.candi.get(g).get(i));
                }
            }
        }
        
         Solution individual = new Solution(this.data);
      chosen = new ArrayList<>();
        for(int g =0;g<data.numberOfGroup;g++){
            ArrayList<Integer> arr = new ArrayList<>();
            for(int i=0;i<data.groupNeed.get(i);i++){
                Random ran = new Random();
                int x = ran.nextInt(candidate.size());
                while(chosen.contains(x)){
                    x = ran.nextInt(500);
                }
                chosen.add(x);
                arr.add(x);
                
            }
            individual.candi.add(arr);
        }
        child.candi = individual.candi;
        return child;
    }
//    
    public void _mutate(Solution individual){
        
            if (Math.random()<0.1){
               individual = problem.generate_solution();
            }
        
    }
    
    public Solution _selection(Population population){
        Solution best = null;
        for(int x=0;x<100;++x)
        {
            int c = (int)(Math.random()*population.population.size());
            if ((best==null)){
                best = population.population.get(c);
            }else{
                if ((crowding_operator(population.population.get(c), best)==1) && (Math.random()<0.1)){
                    best = population.population.get(c);
                }
            }
        }
        return best;
    }
    
}
