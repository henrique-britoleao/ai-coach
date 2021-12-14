from dataclasses import dataclass

@dataclass
class Counter:
    
    valid_reps: int = 0 
    not_valid_reps: int = 0 # TODO: implement detection of bad reps
    
    def add_rep(self):
        self.valid_reps += 1
    
    def flush(self):
        valid_reps = self.valid_reps
        not_valid_reps = self.not_valid_reps
        
        self.valid_reps, self.not_valid_reps = 0, 0 
        
        return valid_reps, not_valid_reps
    
    def __str__(self):
        return f"Counter of reps: {self.valid_reps}"