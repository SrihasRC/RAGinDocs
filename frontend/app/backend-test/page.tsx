"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
// import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Upload, 
  FileText, 
  Scissors, 
  Brain, 
  Database, 
  MessageSquare, 
  CheckCircle, 
  XCircle,
  Loader2,
//   Download,
  Eye,
  Search
} from "lucide-react";

interface UploadResponse {
  file_id: string;
  filename: string;
  file_size: number;
  chunks_created: number;
  embeddings_generated: number;
  processing_time?: number; // Make optional since it might be undefined
  extracted_text: string;
  chunks: Array<{
    chunk_id: string;
    text: string;
    chunk_index: number;
    metadata: Record<string, unknown>;
  }>;
  embeddings_sample: number[][];
}

interface QueryResponse {
  query: string;
  timestamp: string;
  status: string;
  results: {
    context: string;
    suggested_answer: string;
    chunks_found: number;
    sources: Array<{
      chunk_id: string;
      source_file: string;
      page_number: number;
      similarity_score: number;
      text_preview: string;
    }>;
    retrieval_stats: {
      top_similarity: number;
      avg_similarity: number;
      context_length: number;
    };
  };
}

interface ServiceStatus {
  rag_service: string;
  vector_store: {
    status: string;
    document_count: number;
    total_chunks: number;
  };
  embedding_service: {
    status: string;
    model_name: string;
    embedding_dim: number;
  };
}

export default function BackendTester() {
  const [activeTab, setActiveTab] = useState("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = "http://localhost:8000";

  // Check service status
  const checkStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/query/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setServiceStatus(data);
    } catch (err) {
      setError(`Status check failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Upload and process document
  const uploadDocument = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    setUploadResponse(null);
    
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      const response = await fetch(`${API_BASE}/pipeline/document`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setUploadResponse(data);
      setActiveTab("processing");
    } catch (err) {
      setError(`Upload failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Query documents
  const queryDocuments = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    setQueryResponse(null);
    
    try {
      const response = await fetch(`${API_BASE}/query/document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: 5,
          similarity_threshold: 0.2,
          include_metadata: true,
        }),
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setQueryResponse(data);
    } catch (err) {
      setError(`Query failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            RAGinDocs Backend Tester
          </h1>
          <p className="text-muted-foreground">
            Test and visualize your RAG pipeline in real-time
          </p>
        </div>

        {/* Status Check */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Button onClick={checkStatus} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Check Status"}
              </Button>
              
              {serviceStatus && (
                <div className="flex gap-4">
                  <Badge variant={serviceStatus.rag_service === "ready" ? "default" : "destructive"}>
                    RAG: {serviceStatus.rag_service}
                  </Badge>
                  <Badge variant={serviceStatus.vector_store.status === "ready" ? "default" : "destructive"}>
                    Vector DB: {serviceStatus.vector_store.status}
                  </Badge>
                  <Badge variant={serviceStatus.embedding_service.status === "ready" ? "default" : "destructive"}>
                    Embeddings: {serviceStatus.embedding_service.status}
                  </Badge>
                </div>
              )}
            </div>
            
            {serviceStatus && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="font-medium">Documents: {serviceStatus.vector_store.document_count}</p>
                  <p className="text-muted-foreground">Total Chunks: {serviceStatus.vector_store.total_chunks}</p>
                </div>
                <div>
                  <p className="font-medium">Model: {serviceStatus.embedding_service.model_name}</p>
                  <p className="text-muted-foreground">Dimensions: {serviceStatus.embedding_service.embedding_dim}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload & Process
            </TabsTrigger>
            <TabsTrigger value="processing" className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Processing Details
            </TabsTrigger>
            <TabsTrigger value="query" className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Query & RAG
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab */}
          <TabsContent value="upload" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Document Upload
                </CardTitle>
                <CardDescription>
                  Upload a document and see the complete processing pipeline
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Select Document</label>
                  <Input
                    type="file"
                    accept=".pdf,.txt,.docx"
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSelectedFile(e.target.files?.[0] || null)}
                  />
                  {selectedFile && (
                    <div className="text-sm text-muted-foreground">
                      Selected: {selectedFile.name} ({formatBytes(selectedFile.size)})
                    </div>
                  )}
                </div>
                
                <Button 
                  onClick={uploadDocument} 
                  disabled={!selectedFile || loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload & Process Document
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Processing Details Tab */}
          <TabsContent value="processing" className="space-y-6">
            {uploadResponse ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Processing Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      Processing Complete
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium">File</p>
                        <p className="text-sm text-muted-foreground">{uploadResponse.filename || 'Unknown'}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Size</p>
                        <p className="text-sm text-muted-foreground">{formatBytes(uploadResponse.file_size || 0)}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Chunks Created</p>
                        <p className="text-sm text-muted-foreground">{uploadResponse.chunks_created || 0}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Processing Time</p>
                        <p className="text-sm text-muted-foreground">{uploadResponse.processing_time?.toFixed(2) ?? 'N/A'}s</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Text Extraction</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Text Chunking</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Embedding Generation</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Vector Storage</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Extracted Text */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      Extracted Text
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64 w-full rounded border p-4">
                      <pre className="text-sm whitespace-pre-wrap">
                        {uploadResponse.extracted_text ? (
                          <>
                            {uploadResponse.extracted_text.substring(0, 2000)}
                            {uploadResponse.extracted_text.length > 2000 && "\\n\\n... (truncated)"}
                          </>
                        ) : (
                          "No text extracted"
                        )}
                      </pre>
                    </ScrollArea>
                  </CardContent>
                </Card>

                {/* Text Chunks */}
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Scissors className="h-5 w-5" />
                      Text Chunks ({uploadResponse.chunks?.length || 0})
                    </CardTitle>
                    <CardDescription>
                      See how your document was split into chunks for processing
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64 w-full">
                      <div className="space-y-4">
                        {uploadResponse.chunks && uploadResponse.chunks.length > 0 ? (
                          <>
                            {uploadResponse.chunks.slice(0, 10).map((chunk) => (
                              <div key={chunk.chunk_id} className="border rounded p-3 space-y-2">
                                <div className="flex items-center justify-between">
                                  <Badge variant="outline">Chunk {chunk.chunk_index + 1}</Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {chunk.text?.length || 0} chars
                                  </span>
                                </div>
                                <p className="text-sm">{chunk.text?.substring(0, 300) || 'No text'}...</p>
                              </div>
                            ))}
                            {uploadResponse.chunks.length > 10 && (
                              <p className="text-sm text-muted-foreground text-center">
                                ... and {uploadResponse.chunks.length - 10} more chunks
                              </p>
                            )}
                          </>
                        ) : (
                          <p className="text-sm text-muted-foreground text-center">No chunks created</p>
                        )}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>

                {/* Embeddings Sample */}
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      Embeddings Sample
                    </CardTitle>
                    <CardDescription>
                      First few embeddings vectors (384 dimensions each)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-32 w-full">
                      <div className="space-y-2">
                        {uploadResponse.embeddings_sample && uploadResponse.embeddings_sample.length > 0 ? (
                          uploadResponse.embeddings_sample.slice(0, 3).map((embedding, index) => (
                            <div key={index} className="text-xs font-mono">
                              <span className="text-muted-foreground">Chunk {index + 1}:</span> [
                              {embedding && embedding.length > 0 ? 
                                embedding.slice(0, 10).map(val => val.toFixed(4)).join(", ") + ", ..." :
                                "No embedding data"
                              }
                              ]
                            </div>
                          ))
                        ) : (
                          <p className="text-xs text-muted-foreground">No embedding samples available</p>
                        )}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-64">
                  <div className="text-center space-y-2">
                    <FileText className="h-12 w-12 mx-auto text-muted-foreground" />
                    <p className="text-muted-foreground">Upload a document first to see processing details</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Query Tab */}
          <TabsContent value="query" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  RAG Query Testing
                </CardTitle>
                <CardDescription>
                  Test your RAG system with natural language queries
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Your Question</label>
                  <Textarea
                    placeholder="What are the key terms in the contract?"
                    value={query}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value)}
                    rows={3}
                  />
                </div>
                
                <Button 
                  onClick={queryDocuments} 
                  disabled={!query.trim() || loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Search className="h-4 w-4 mr-2" />
                      Query Documents
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Query Results */}
            {queryResponse && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Answer */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      AI Response
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="border rounded p-4 bg-muted/50">
                        <p className="text-sm">{queryResponse.results.suggested_answer}</p>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="font-medium">Chunks Found</p>
                          <p className="text-muted-foreground">{queryResponse.results.chunks_found}</p>
                        </div>
                        <div>
                          <p className="font-medium">Top Similarity</p>
                          <p className="text-muted-foreground">
                            {(queryResponse.results.retrieval_stats.top_similarity * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="font-medium">Context Length</p>
                          <p className="text-muted-foreground">
                            {queryResponse.results.retrieval_stats.context_length} chars
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Retrieved Context */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      Retrieved Context
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64 w-full rounded border p-4">
                      <pre className="text-sm whitespace-pre-wrap">
                        {queryResponse.results.context}
                      </pre>
                    </ScrollArea>
                  </CardContent>
                </Card>

                {/* Sources */}
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" />
                      Source Chunks
                    </CardTitle>
                    <CardDescription>
                      Document chunks that contributed to the answer
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {queryResponse.results.sources.map((source) => (
                        <div key={source.chunk_id} className="border rounded p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">
                                Similarity: {(source.similarity_score * 100).toFixed(1)}%
                              </Badge>
                              <span className="text-sm text-muted-foreground">
                                {source.source_file} - Page {source.page_number}
                              </span>
                            </div>
                          </div>
                          <p className="text-sm">{source.text_preview}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
