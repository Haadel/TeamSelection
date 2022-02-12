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
public class Solution {

    ArrayList<ArrayList<Integer>> candi = new ArrayList<>();
    int rank;
    double fitness;
    public double crowding_distance = 0;
    public double[] objectives;

    public int domination_count;
    public ArrayList<Solution> dominated_solution;

    public Solution(Data data) {
        objectives = new double[data.numberOfGroup * 2];
    }

    public boolean dominates(Object o) {
//        if (this==o){
//            return false;
//        }

        if (!(o instanceof Solution)) {
            return false;
        }

        Solution other = (Solution) o;
        boolean and_condition = true;
        boolean or_condition = false;
        for (int i = 0; i < 6; i++) {
            and_condition = and_condition && (this.objectives[i] <= other.objectives[i]);
            or_condition = or_condition || (this.objectives[i] < other.objectives[i]);
        }
        return (and_condition && or_condition);
    }

    public double callFitness(Data data) {
        double fitness = 0;
        Double[] deep = new Double[data.numberOfGroup];
        Double[] wise = new Double[data.numberOfGroup];
        for (int g = 0; g < data.numberOfGroup; g++) {
            double deep_obj = 0;
            double wise_obj = 0;
            for (int c = 0; c < data.groupNeed.get(g); c++) {
                for (int s = 0; s < data.numberOfSkill; s++) {
                    deep_obj += data.neededSkill[g][s] * data.candidateSkill[this.candi.get(g).get(c)][s];
                    wise_obj += Double.min(1.0, data.neededSkill[g][s] * data.candidateSkill[this.candi.get(g).get(c)][s]);

                }

            }
            deep[g] = deep_obj;
            wise[g] = wise_obj;

        }
        System.out.println(deep[1]);
        for (int g = 0; g < data.numberOfGroup; g++) {
            fitness +=data.dw[g] * Math.pow( ((deep[g] - data.maxPointDeep[g]) / (data.maxPointDeep[g] - data.minPointDeep[g])), 2);
        }
        fitness = Math.sqrt(fitness);
        double fit = 0;
        for (int g = 0; g < data.numberOfGroup; g++) {
            fit +=data.ww[g] * Math.pow( ((wise[g] - data.maxPointDeep[g]) / (data.maxPointDeep[g] - data.minPointDeep[g])), 2);
        }
        fit = Math.sqrt(fit);
        fitness+=fit;
        return fitness/2;
    }
}
