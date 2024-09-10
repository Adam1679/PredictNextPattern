# PredictNextPattern
Tokenize the Stock Price and then apply Generative LLM.
## Tokenization
### Find Common Single Bar Pattern
- 
### Then Cross two bar pattern (Up/Down) for Time Series related.
### Run BPE


### Learnings

1. No scaling law is detected,
2. varying time window helps?
3. T5 better than GPT2. Decoder-Only will overfit.
4. Lion Optimizer seems good in terms of validation loss.
5. z_reg doesn’t hurt or help.
6. predict higher_high and lower_low is more efficient. You will get ~68 precision.
7. multi-task learning helps?
8. longer context window helps?
9. predict the price is not working. It's just random prediction.