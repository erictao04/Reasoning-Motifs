import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { MotifCardDetailPage } from "./pages/MotifCardDetailPage";
import { MotifCardExplorerPage } from "./pages/MotifCardExplorerPage";
import { PipelineFigurePage } from "./pages/PipelineFigurePage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<MotifCardExplorerPage />} />
        <Route path="/pipeline" element={<PipelineFigurePage />} />
        <Route path="/questions/:questionId" element={<MotifCardDetailPage />} />
        <Route path="/motif-cards" element={<Navigate to="/" replace />} />
        <Route path="/motif-cards/:questionId" element={<Navigate to="/" replace />} />
        <Route path="/questions" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
