import MLX
import MLXRandom
import MLXNN
import MLXOptimizers
import FastVLM  // Your MLX-based VLM

class MLXVLMModel: ObservableObject {
    @Published var output = ""
    @Published var running = false
    
    private var modelContainer: ModelContainer?
    
    func loadModel() async throws {
        // Load your converted MLX VLM model
        let modelPath = Bundle.main.path(forResource: "custom-vlm-mlx", ofType: nil)!
        modelContainer = try await ModelContainer.load(path: modelPath)
    }
    
    func generate(_ userInput: UserInput) async {
        guard let container = modelContainer else { return }
        
        running = true
        
        do {
            let result = try await container.perform { context in
                return try MLXLMCommon.generate(
                    input: try await context.processor.prepare(input: userInput),
                    parameters: GenerateParameters(),
                    context: context
                ) { tokens in
                    let text = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in
                        self.output = text
                    }
                    return tokens.count >= 512 ? .stop : .more
                }
            }
            
            await MainActor.run {
                self.output = result.output
            }
        } catch {
            await MainActor.run {
                self.output = "Error: \(error)"
            }
        }
        
        running = false
    }
}