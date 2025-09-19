"use client";

import { useState } from "react";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";

export default function Home() {
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);

  return (
    <div className="h-screen bg-background flex">
      {/* Left Side - Documents */}
      <div className="w-96 border-r border-border bg-sidebar">
        <DocumentSidebar 
          selectedDocument={selectedDocument}
          onDocumentSelect={setSelectedDocument}
        />
      </div>
      
      {/* Right Side - Chat */}
      <div className="flex-1 flex flex-col">
        <ChatInterface selectedDocument={selectedDocument} />
      </div>
    </div>
  );
}
