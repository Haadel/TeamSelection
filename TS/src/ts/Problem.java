/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author ACER
 */
public class Problem {
    private Data data;
    
    public Problem(Data data){
        this.data = data;
    }
    
    public Solution generate_solution(){
        Solution individual = new Solution(this.data);
        individual.candi = new ArrayList<>();
       ArrayList<Integer> chosen = new ArrayList<>();
        for(int g =0;g<data.numberOfGroup;g++){
            ArrayList<Integer> arr = new ArrayList<>();
            for(int i=0;i<data.groupNeed.get(i);i++){
                Random ran = new Random();
                int x = ran.nextInt(500);
                while(chosen.contains(x)){
                    x = ran.nextInt(500);
                }
                chosen.add(x);
                arr.add(x);
                
            }
            individual.candi.add(arr);
        }
        
        return individual;
    }
    
    public void cal_objectives(Solution individual){
        individual.objectives = new double[6];
       for(int g =0;g<data.numberOfGroup;g++){
         
            double deep_obj = 0;
            double wise_obj = 0;
            for (int c = 0; c < data.groupNeed.get(g); c++) {
                for (int s = 0; s < data.numberOfSkill; s++) {
                    deep_obj += data.neededSkill[g][s] * data.candidateSkill[individual.candi.get(g).get(c)][s];
                    wise_obj += Double.min(1.0, data.neededSkill[g][s] * data.candidateSkill[individual.candi.get(g).get(c)][s]);

                }

            }
            individual.objectives[g*2]=Math.sqrt(wise_obj);
            individual.objectives[g*2+1]=wise_obj;
            
           

        
       }
    }
    
    public boolean check_constraint(Solution individual){
       return true;
    }
}
