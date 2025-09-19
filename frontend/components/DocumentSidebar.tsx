"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Upload, FileText, File, Clock, CheckCircle } from "lucide-react";

interface Document {
  id: string;
  file_name: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  page_count: number;
  processing_status: string;
}

interface DocumentSidebarProps {
  selectedDocument: string | null;
  onDocumentSelect: (docId: string | null) => void;
}

export function DocumentSidebar({ selectedDocument, onDocumentSelect }: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  // Fetch documents from backend
  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://localhost:8000/documents/list");
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", "frontend-user");

    try {
      const response = await fetch("http://localhost:8000/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        await fetchDocuments(); // Refresh the list
      } else {
        console.error("Upload failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
    } finally {
      setUploading(false);
      // Reset input
      if (event.target) {
        event.target.value = "";
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getFileIcon = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case ".pdf":
        return <FileText className="w-4 h-4 text-red-500" />;
      case ".docx":
      case ".doc":
        return <FileText className="w-4 h-4 text-blue-500" />;
      case ".txt":
        return <File className="w-4 h-4 text-gray-500" />;
      default:
        return <File className="w-4 h-4" />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <FileText className="w-4 h-4 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-sidebar-foreground">RAGinDocs</h1>
            <p className="text-xs text-muted-foreground">Document Intelligence</p>
          </div>
        </div>

        {/* Upload Button */}
        <div className="relative">
          <input
            type="file"
            accept=".pdf,.docx,.doc,.txt"
            onChange={handleFileUpload}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={uploading}
          />
          <Button 
            className="w-full" 
            disabled={uploading}
          >
            <Upload className="w-4 h-4 mr-2" />
            {uploading ? "Uploading..." : "Upload Document"}
          </Button>
        </div>
      </div>

      {/* Documents List */}
      <div className="flex-1 p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-sidebar-foreground">
            Documents ({documents.length})
          </h2>
        </div>

        <ScrollArea className="h-full">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-sm text-muted-foreground">Loading documents...</div>
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FileText className="w-12 h-12 text-muted-foreground mb-3" />
              <p className="text-sm text-muted-foreground mb-2">No documents uploaded</p>
              <p className="text-xs text-muted-foreground">Upload your first document to get started</p>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <Card
                  key={doc.id}
                  className={`cursor-pointer transition-colors hover:bg-sidebar-accent ${
                    selectedDocument === doc.id ? "bg-sidebar-accent border-primary" : ""
                  }`}
                  onClick={() => 
                    onDocumentSelect(selectedDocument === doc.id ? null : doc.id)
                  }
                >
                  <CardContent className="p-3">
                    <div className="flex items-start gap-3">
                      {getFileIcon(doc.file_type)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm font-medium text-sidebar-foreground truncate">
                            {doc.file_name}
                          </p>
                          {doc.processing_status === "completed" ? (
                            <CheckCircle className="w-3 h-3 text-green-500 flex-shrink-0" />
                          ) : (
                            <Clock className="w-3 h-3 text-yellow-500 flex-shrink-0" />
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>{formatFileSize(doc.file_size)}</span>
                          <span>•</span>
                          <span>{formatDate(doc.upload_date)}</span>
                        </div>
                        {doc.page_count > 0 && (
                          <Badge variant="secondary" className="mt-1 text-xs">
                            {doc.page_count} pages
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
