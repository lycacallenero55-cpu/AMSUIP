import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { BookOpen, Construction } from "lucide-react";

const Subjects = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4 space-y-4">
          {/* Header */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div>
              <h1 className="text-lg font-bold text-education-navy">SUBJECTS</h1>
              <p className="text-sm text-muted-foreground">
                Manage and organize academic subjects and courses
              </p>
            </div>
          </div>

          {/* Space between title and first card */}
          <div className="mb-16"></div>

          {/* Under Development Message */}
          <Card className="bg-gradient-card border border-gray-200 shadow-card">
            <CardContent className="p-12">
              <div className="flex flex-col items-center justify-center text-center space-y-4">
                <div className="p-4 rounded-full bg-gradient-primary/10">
                  <Construction className="h-12 w-12 text-primary" />
                </div>
                <div className="space-y-2">
                  <h2 className="text-xl font-semibold text-education-navy">
                    This page is under development
                  </h2>
                  <p className="text-sm text-muted-foreground max-w-md">
                    We're working hard to bring you a comprehensive subjects management system. 
                    This feature will allow you to organize and manage academic subjects and courses.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <BookOpen className="w-4 h-4" />
                  <span>Coming soon</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Subjects;