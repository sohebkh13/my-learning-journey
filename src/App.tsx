import React, { useState } from 'react';
import { Brain, Database, Bot, Home, BookOpen, Code, Zap, Users, GitBranch, BarChart3, Cpu, Network, FileText, Play, CheckCircle } from 'lucide-react';
import JupyterNotebook from './components/JupyterNotebook';

interface Topic {
  id: string;
  title: string;
  completed?: boolean;
}

interface Section {
  id: string;
  title: string;
  icon: React.ReactNode;
  color: string;
  topics: Topic[];
}

const sections: Section[] = [
  {
    id: 'llm-engineering',
    title: 'LLM Engineering',
    icon: <Brain className="w-5 h-5" />,
    color: 'bg-blue-500',
    topics: [
      { id: 'intro-llm', title: 'Introduction to LLMs', completed: true },
      { id: 'transformer-architecture', title: 'Transformer Architecture' },
      { id: 'prompt-engineering', title: 'Prompt Engineering', completed: true },
      { id: 'fine-tuning', title: 'Fine-tuning Techniques' },
      { id: 'rag-systems', title: 'RAG Systems' },
      { id: 'vector-databases', title: 'Vector Databases' },
      { id: 'llm-evaluation', title: 'LLM Evaluation' },
      { id: 'deployment-scaling', title: 'Deployment & Scaling' },
      { id: 'llm-security', title: 'LLM Security' },
      { id: 'cost-optimization', title: 'Cost Optimization' }
    ]
  },
  {
    id: 'agentic-ai',
    title: 'Agentic AI',
    icon: <Bot className="w-5 h-5" />,
    color: 'bg-purple-500',
    topics: [
      { id: 'agent-fundamentals', title: 'Agent Fundamentals' },
      { id: 'multi-agent-systems', title: 'Multi-Agent Systems' },
      { id: 'tool-calling', title: 'Tool Calling & Function Use' },
      { id: 'planning-reasoning', title: 'Planning & Reasoning' },
      { id: 'memory-systems', title: 'Memory Systems' },
      { id: 'agent-frameworks', title: 'Agent Frameworks' },
      { id: 'human-ai-interaction', title: 'Human-AI Interaction' },
      { id: 'agent-evaluation', title: 'Agent Evaluation' },
      { id: 'ethical-considerations', title: 'Ethical Considerations' }
    ]
  },
  {
    id: 'data-engineering',
    title: 'Data Engineering',
    icon: <Database className="w-5 h-5" />,
    color: 'bg-green-500',
    topics: [
      { id: 'data-pipelines', title: 'Data Pipelines', completed: true },
      { id: 'etl-elt', title: 'ETL vs ELT' },
      { id: 'data-warehousing', title: 'Data Warehousing' },
      { id: 'stream-processing', title: 'Stream Processing' },
      { id: 'data-quality', title: 'Data Quality & Validation' },
      { id: 'orchestration', title: 'Workflow Orchestration' },
      { id: 'cloud-platforms', title: 'Cloud Data Platforms' },
      { id: 'data-governance', title: 'Data Governance' },
      { id: 'monitoring-observability', title: 'Monitoring & Observability' }
    ]
  }
];

const topicContent: Record<string, { title: string; content: string; example?: string }> = {
  'intro-llm': {
    title: 'Introduction to Large Language Models',
    content: `Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text. They represent a breakthrough in natural language processing and have revolutionized how we interact with AI.

Key characteristics of LLMs:
• Trained on billions of parameters
• Can perform various language tasks without specific training
• Exhibit emergent abilities at scale
• Foundation for many AI applications

Popular LLMs include GPT-4, Claude, Gemini, and LLaMA.`,
    example: `# Example: Simple LLM interaction
prompt = "Explain quantum computing in simple terms"
response = llm.generate(prompt)
print(response)`
  },
  'fine-tuning': {
    title: 'Fine-tuning Techniques',
    content: `Fine-tuning is the process of adapting a pre-trained language model to perform better on specific tasks or domains. This technique allows you to leverage the knowledge of large models while customizing them for your specific use case.

Key concepts:
• Transfer Learning: Using pre-trained weights as starting point
• Task-specific adaptation: Modifying model behavior for specific tasks
• Parameter-efficient methods: LoRA, Adapters, Prompt tuning
• Full fine-tuning vs. Parameter-efficient fine-tuning

Popular frameworks: Hugging Face Transformers, Axolotl, Unsloth`,
    example: `# Fine-tuning with Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Fine-tune on your dataset
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results"),
    train_dataset=train_dataset
)
trainer.train()`
  },
  'prompt-engineering': {
    title: 'Prompt Engineering',
    content: `Prompt engineering is the art and science of crafting effective inputs to get desired outputs from LLMs. It's a crucial skill for maximizing LLM performance.

Key techniques:
• Clear and specific instructions
• Few-shot learning with examples
• Chain-of-thought prompting
• Role-based prompting
• Template-based approaches

Best practices:
• Be specific about the desired format
• Provide context and examples
• Use delimiters to separate sections
• Iterate and refine prompts`,
    example: `# Example: Chain-of-thought prompting
prompt = """
Solve this step by step:
What is 15% of 240?

Step 1: Convert percentage to decimal
Step 2: Multiply by the number
Step 3: State the final answer
"""
`
  },
  'data-pipelines': {
    title: 'Data Pipelines',
    content: `Data pipelines are automated workflows that move and transform data from source systems to destination systems. They are the backbone of modern data architecture.

Components of a data pipeline:
• Data Sources (databases, APIs, files)
• Ingestion layer
• Processing/Transformation layer
• Storage layer
• Monitoring and alerting

Types of pipelines:
• Batch processing
• Real-time/streaming
• Hybrid approaches

Popular tools: Apache Airflow, Prefect, Dagster, Apache Kafka`,
    example: `# Example: Simple data pipeline with Python
import pandas as pd
from datetime import datetime

def extract_data():
    # Extract from source
    return pd.read_csv('source_data.csv')

def transform_data(df):
    # Clean and transform
    df['processed_date'] = datetime.now()
    return df.dropna()

def load_data(df):
    # Load to destination
    df.to_sql('processed_table', connection)

# Pipeline execution
data = extract_data()
transformed = transform_data(data)
load_data(transformed)`
  }
};

function App() {
  const [activeSection, setActiveSection] = useState<string>('home');
  const [activeTopic, setActiveTopic] = useState<string>('');

  const currentSection = sections.find(s => s.id === activeSection);
  const currentTopic = topicContent[activeTopic];

  const renderHome = () => (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">My Learning Journey</h1>
        <p className="text-xl text-gray-600 mb-8">
          Documenting my path through LLM Engineering, Agentic AI, and Data Engineering
        </p>
        <div className="flex justify-center space-x-4 text-sm text-gray-500">
          <span className="flex items-center"><CheckCircle className="w-4 h-4 mr-1 text-green-500" /> Completed</span>
          <span className="flex items-center"><Play className="w-4 h-4 mr-1 text-blue-500" /> In Progress</span>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        {sections.map((section) => {
          const completedTopics = section.topics.filter(t => t.completed).length;
          const totalTopics = section.topics.length;
          const progress = (completedTopics / totalTopics) * 100;

          return (
            <div
              key={section.id}
              className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer border-l-4"
              style={{ borderLeftColor: section.color.replace('bg-', '').replace('-500', '') === 'blue' ? '#3b82f6' : section.color.replace('bg-', '').replace('-500', '') === 'purple' ? '#8b5cf6' : '#10b981' }}
              onClick={() => setActiveSection(section.id)}
            >
              <div className="flex items-center mb-4">
                <div className={`${section.color} text-white p-3 rounded-lg mr-4`}>
                  {section.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-800">{section.title}</h3>
              </div>
              <p className="text-gray-600 mb-4">
                {section.id === 'llm-engineering' && 'Master the fundamentals of Large Language Models, from architecture to deployment.'}
                {section.id === 'agentic-ai' && 'Explore autonomous AI agents and multi-agent systems for complex problem solving.'}
                {section.id === 'data-engineering' && 'Build robust data pipelines and infrastructure for AI applications.'}
              </p>
              <div className="mb-2">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Progress</span>
                  <span>{completedTopics}/{totalTopics}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${section.color}`}
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">About This Journey</h2>
        <p className="text-gray-700 leading-relaxed">
          This website documents my learning journey in the exciting fields of AI and data engineering. 
          Each section contains structured learning paths, practical examples, and real-world applications. 
          The goal is to build comprehensive knowledge that bridges theory with practice, enabling the 
          development of production-ready AI systems.
        </p>
      </div>
    </div>
  );

  const renderSectionContent = () => {
    if (!currentSection) return null;

    return (
      <div className="flex h-full">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-lg">
          <div className="p-4 border-b">
            <div className="flex items-center">
              <div className={`${currentSection.color} text-white p-2 rounded mr-3`}>
                {currentSection.icon}
              </div>
              <h2 className="font-semibold text-gray-800">{currentSection.title}</h2>
            </div>
          </div>
          <nav className="p-4">
            {currentSection.topics.map((topic) => (
              <button
                key={topic.id}
                onClick={() => setActiveTopic(topic.id)}
                className={`w-full text-left p-3 rounded-lg mb-2 transition-colors flex items-center justify-between ${
                  activeTopic === topic.id
                    ? 'bg-blue-100 text-blue-800 border-l-4 border-blue-500'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <span className="text-sm">{topic.title}</span>
                {topic.completed && <CheckCircle className="w-4 h-4 text-green-500" />}
              </button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-8 overflow-y-auto">
          {currentTopic ? (
            <div className="max-w-4xl">
              <h1 className="text-3xl font-bold text-gray-800 mb-6">{currentTopic.title}</h1>
              <div className="prose prose-lg max-w-none">
                <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                  <pre className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                    {currentTopic.content}
                  </pre>
                </div>
                {currentTopic.example && (
                  <div className="bg-gray-900 rounded-lg p-6 mb-6">
                    <div className="flex items-center mb-3">
                      <Code className="w-5 h-5 text-green-400 mr-2" />
                      <span className="text-green-400 font-medium">Example</span>
                    </div>
                    <pre className="text-green-300 text-sm overflow-x-auto">
                      <code>{currentTopic.example}</code>
                    </pre>
                  </div>
                )}
                
                {/* Jupyter Notebook Example for Fine-tuning */}
                {activeTopic === 'fine-tuning' && (
                  <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-gray-800 mb-4">Practical Example: Fine-tuning with LoRA</h2>
                    <JupyterNotebook
                      title="fine_tuning_lora_example.ipynb"
                      cells={[
                        {
                          type: 'markdown',
                          content: '<h2>Fine-tuning LLaMA with LoRA</h2><p>This notebook demonstrates how to fine-tune a LLaMA model using Low-Rank Adaptation (LoRA) for a custom dataset.</p>'
                        },
                        {
                          type: 'code',
                          content: `# Install required packages
!pip install transformers peft datasets accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset`,
                          executionCount: 1,
                          output: `Collecting transformers
Successfully installed transformers-4.35.2 peft-0.6.2 datasets-2.14.6`
                        },
                        {
                          type: 'code',
                          content: `# Load the base model and tokenizer
model_name = "microsoft/DialoGPT-small"  # Using smaller model for demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Model loaded: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")`,
                          executionCount: 2,
                          output: `Model loaded: microsoft/DialoGPT-small
Model parameters: 117,184,640`
                        },
                        {
                          type: 'code',
                          content: `# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]  # Target attention layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()`,
                          executionCount: 3,
                          output: `trainable params: 294,912 || all params: 117,479,552 || trainable%: 0.25`
                        },
                        {
                          type: 'code',
                          content: `# Load and prepare dataset
dataset = load_dataset("json", data_files="custom_conversations.json")

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(f"Dataset size: {len(tokenized_dataset['train'])}")`,
                          executionCount: 4,
                          output: `Dataset size: 1,250`
                        },
                        {
                          type: 'code',
                          content: `# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-finetuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

print("Training configuration set up successfully!")`,
                          executionCount: 5,
                          output: `Training configuration set up successfully!`
                        },
                        {
                          type: 'code',
                          content: `# Start training (this would take time in real scenario)
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# trainer.train()  # Commented out for demo
print("Trainer initialized. Ready to start fine-tuning!")`,
                          executionCount: 6,
                          output: `Trainer initialized. Ready to start fine-tuning!`
                        },
                        {
                          type: 'markdown',
                          content: '<h3>Results</h3><p>After training, the model will be adapted to your specific dataset while maintaining the general knowledge from the pre-trained model. LoRA allows us to fine-tune with only 0.25% of the original parameters!</p>'
                        }
                      ]}
                    />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <BookOpen className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p className="text-xl">Select a topic to start learning</p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <button
                onClick={() => {
                  setActiveSection('home');
                  setActiveTopic('');
                }}
                className="flex items-center space-x-2 text-xl font-bold text-gray-800 hover:text-blue-600 transition-colors"
              >
                <Zap className="w-6 h-6 text-blue-500" />
                <span>Learning Hub</span>
              </button>
            </div>
            <nav className="flex space-x-8">
              <button
                onClick={() => {
                  setActiveSection('home');
                  setActiveTopic('');
                }}
                className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeSection === 'home'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Home className="w-4 h-4" />
                <span>Home</span>
              </button>
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => {
                    setActiveSection(section.id);
                    setActiveTopic('');
                  }}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeSection === section.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  {section.icon}
                  <span className="hidden md:inline">{section.title}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="h-[calc(100vh-4rem)]">
        {activeSection === 'home' ? (
          <div className="p-8">
            {renderHome()}
          </div>
        ) : (
          renderSectionContent()
        )}
      </main>
    </div>
  );
}

export default App;